#!/usr/bin/env python3
from __future__ import annotations

import argparse, gc, importlib.util, json, math, os, re, shutil, time, unicodedata
from pathlib import Path
from collections import Counter

import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm

HERE=Path(__file__).resolve().parent

# Exact PT15 Dynamic-k implementation captured from the user's workstation.
spec=importlib.util.spec_from_file_location("dynk",HERE/"01_risk_adaptive_dynamic_k.py")
D=importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
B=D.B

PT15_CKPT=Path("/data/smgreen1/voxtral_moe_v1/dynamic_k_exact_budget_recovery_v1_0/GLOBAL_BEST/moe_trainable_state.pt")
OUTROOT=Path("/data/smgreen1/voxtral_moe_v1/pt57g_asr_specialized_dynamic_k_recovery_v1_0")

# Preserve the strongest certified PT15 Dynamic-k concept: uncertainty-only.
D.ALPHA_U=1.0
D.BETA_C=0.0

TARGET_K1=0.18
TARGET_K3=0.08
TARGET_AVG=1.90
CERT_MIN=1.80
CERT_MAX=1.98

def load_pt15_state(model):
    state=torch.load(PT15_CKPT,map_location="cpu")
    named=dict(model.named_parameters())
    loaded=[]; missing=[]; shape_bad=[]
    for n,t in state.items():
        if n not in named:
            missing.append(n); continue
        if tuple(named[n].shape)!=tuple(t.shape):
            shape_bad.append((n,tuple(named[n].shape),tuple(t.shape))); continue
        named[n].data.copy_(t.to(device=named[n].device,dtype=named[n].dtype))
        loaded.append(n)
    print("PT15_STATE_TENSORS:",len(state))
    print("PT15_LOADED_TENSORS:",len(loaded))
    print("PT15_UNMATCHED:",len(missing))
    print("PT15_SHAPE_BAD:",len(shape_bad))
    if len(state)!=624 or missing or shape_bad or len(loaded)!=624:
        raise RuntimeError("PT15 state did not load exactly as 624 MoE tensors.")

def clean_metric(s):
    return re.sub(r"\s+"," ","" if s is None else str(s)).strip()

def unicode_normalize_for_diagnostic(s):
    # Diagnostic only; never used to change the frozen references.
    s=unicodedata.normalize("NFKC",clean_metric(s)).lower()
    s="".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return re.sub(r"\s+"," ",s).strip()

def align_counts(ref,hyp):
    r=ref.split(); h=hyp.split()
    nr,nh=len(r),len(h)
    dp=[[None]*(nh+1) for _ in range(nr+1)]
    dp[0][0]=(0,0,0,0) # total,S,D,I
    for i in range(1,nr+1): dp[i][0]=(i,0,i,0)
    for j in range(1,nh+1): dp[0][j]=(j,0,0,j)
    for i in range(1,nr+1):
        for j in range(1,nh+1):
            if r[i-1]==h[j-1]:
                dp[i][j]=dp[i-1][j-1]
            else:
                a=dp[i-1][j-1]; sub=(a[0]+1,a[1]+1,a[2],a[3])
                a=dp[i-1][j]; dele=(a[0]+1,a[1],a[2]+1,a[3])
                a=dp[i][j-1]; ins=(a[0]+1,a[1],a[2],a[3]+1)
                dp[i][j]=min((sub,dele,ins),key=lambda x:(x[0],x[2],x[3],x[1]))
    _,s,d,i=dp[nr][nh]
    return s,d,i

@torch.no_grad()
def infer_cfg(model,processor,audio_path,device,max_new_tokens=256,repetition_penalty=1.3,no_repeat_ngram_size=3):
    model.eval()
    x=B.prompt_inputs(processor,audio_path,device)
    plen=x["input_ids"].shape[1]
    kwargs=dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        repetition_penalty=float(repetition_penalty),
    )
    if int(no_repeat_ngram_size)>0:
        kwargs["no_repeat_ngram_size"]=int(no_repeat_ngram_size)
    t0=time.time()
    out=model.generate(**x,**kwargs)
    sec=time.time()-t0
    gen=out[:,plen:]
    txt=processor.batch_decode(gen,skip_special_tokens=True)[0]
    return B.clean(txt),sec,int(gen.shape[1]),bool(gen.shape[1]>=int(max_new_tokens))

@torch.no_grad()
def evaluate(model,processor,df,device,label,out_csv,max_new_tokens=256,repetition_penalty=1.3,no_repeat_ngram_size=3):
    D.reset_stats(model)
    rows=[]; refs=[]; hyps=[]
    S=Dels=Ins=0
    print(f"\n=== {label} ===")
    for i,(_,r) in enumerate(df.iterrows(),1):
        g=B.clean_transcript(r["asr_target_clean"])
        p,sec,gtok,hit=infer_cfg(
            model,processor,r["resolved_audio_path"],device,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        rw=B.row_wer(g,p)
        s,d,ins=align_counts(g,p)
        S+=s; Dels+=d; Ins+=ins
        refs.append(g); hyps.append(p)
        gn=B.clean_transcript(unicode_normalize_for_diagnostic(g))
        pn=B.clean_transcript(unicode_normalize_for_diagnostic(p))
        rows.append({
            "audio_name":r["audio_name"],
            "reference_transcript":g,
            "predicted_transcript":p,
            "row_wer_ehsan_style":rw,
            "row_cer":B.row_cer(g,p),
            "token_f1":B.token_f1(g,p),
            "unicode_normalized_row_wer_diagnostic":B.row_wer(gn,pn),
            "substitutions":s,"deletions":d,"insertions":ins,
            "runtime_sec":sec,
            "generated_tokens":gtok,
            "hit_max_new_tokens":int(hit),
            "blank":int(not p),
            "abnormally_short_vs_ref_DIAGNOSTIC":int(len(p.split()) < 0.35*max(1,len(g.split()))),
        })
        if i<=5 or i%25==0 or i==len(df):
            mean=sum(x["row_wer_ehsan_style"] for x in rows)/len(rows)
            print(f"[{i:04d}/{len(df)}] rowWER={rw:.4f} mean={mean:.4f} blank={int(not p)} toks={gtok}")
    ddf=pd.DataFrame(rows)
    out_csv=Path(out_csv);out_csv.parent.mkdir(parents=True,exist_ok=True)
    ddf.to_csv(out_csv,index=False)
    norm_refs=[unicode_normalize_for_diagnostic(x) for x in refs]
    norm_hyps=[unicode_normalize_for_diagnostic(x) for x in hyps]
    m={
        "n_rows":len(ddf),
        "mean_row_WER_Ehsan_style":float(ddf.row_wer_ehsan_style.mean()),
        "mean_row_CER":float(ddf.row_cer.mean()),
        "corpus_WER":float(B.corpus_wer(refs,hyps)),
        "mean_TokenF1":float(ddf.token_f1.mean()),
        "unicode_normalized_mean_row_WER_DIAGNOSTIC":float(ddf.unicode_normalized_row_wer_diagnostic.mean()),
        "unicode_normalized_corpus_WER_DIAGNOSTIC":float(B.corpus_wer(norm_refs,norm_hyps)),
        "mean_runtime_sec":float(ddf.runtime_sec.mean()),
        "blank_predictions":int(ddf.blank.sum()),
        "hit_max_new_tokens_rows":int(ddf.hit_max_new_tokens.sum()),
        "abnormally_short_rows_DIAGNOSTIC":int(ddf.abnormally_short_vs_ref_DIAGNOSTIC.sum()),
        "substitutions":int(S),"deletions":int(Dels),"insertions":int(Ins),
        **D.route_summary(model),
    }
    print("METRICS:",json.dumps(m,indent=2))
    return m

def choose_rows(df,n,seed,exclude=None):
    d=df
    if exclude:
        d=d[~d["audio_name"].astype(str).isin(set(exclude))].copy()
    return d.sample(n=min(int(n),len(d)),random_state=seed).reset_index(drop=True)

@torch.no_grad()
def collect_hist_and_routes(model,processor,df,device,dynamic,label):
    D.DYNAMIC_ENABLED=bool(dynamic)
    D.CURRENT_CRIT_TARGET=None
    D.COLLECT_RISK=False; D.RISK_SAMPLES=[]
    D.COLLECT_RISK_HIST=True; D.RISK_HIST=None
    D.reset_stats(model)
    pb=tqdm(df.iterrows(),total=len(df),desc=label,dynamic_ncols=True,mininterval=5)
    for _,r in pb:
        _p,_sec=B.infer_one(model,processor,r["resolved_audio_path"],device,max_new_tokens=256)
    pb.close()
    D.COLLECT_RISK_HIST=False
    if D.RISK_HIST is None: raise RuntimeError("Risk histogram was not collected.")
    return D.RISK_HIST.detach().float().cpu(),D.route_summary(model)

def hist_quantile(hist,q):
    total=float(hist.sum())
    if total<=0: raise RuntimeError("Empty risk histogram.")
    c=torch.cumsum(hist,dim=0)
    target=torch.tensor(float(q)*total,dtype=c.dtype)
    idx=int(torch.searchsorted(c,target).item())
    idx=max(0,min(idx,len(hist)-1))
    return (idx+0.5)/len(hist)

def thresholds_from_hist(hist):
    D.TAU1=float(hist_quantile(hist,TARGET_K1))
    D.TAU2=float(hist_quantile(hist,1.0-TARGET_K3))
    if not D.TAU1<D.TAU2:
        raise RuntimeError(f"Invalid Dynamic-k thresholds: {D.TAU1}, {D.TAU2}")

def exact_budget_calibrate(model,processor,train_df,device,cal_rows=64,cert_rows=24,iters=2,seed_base=5700):
    cal=choose_rows(train_df,cal_rows,seed_base+1)
    names=cal["audio_name"].astype(str).tolist()
    cert=choose_rows(train_df,cert_rows,seed_base+2,exclude=names)
    hist,r0=collect_hist_and_routes(model,processor,cal,device,False,"PT57G CAL fixed-k2")
    thresholds_from_hist(hist)
    history=[{"stage":"fixed_k2","tau1":D.TAU1,"tau2":D.TAU2,"routes":r0}]
    for i in range(1,iters+1):
        hist,rr=collect_hist_and_routes(model,processor,cal,device,True,f"PT57G CAL dynamic iter{i}")
        thresholds_from_hist(hist)
        history.append({"stage":f"dynamic_iter_{i}","tau1":D.TAU1,"tau2":D.TAU2,"routes_before_update":rr})
    _h,cert_routes=collect_hist_and_routes(model,processor,cert,device,True,"PT57G TRAIN budget cert")
    passed=bool(CERT_MIN<=cert_routes["average_routed_k"]<=CERT_MAX)
    result={
        "target_k1_fraction":TARGET_K1,
        "target_k2_fraction":1-TARGET_K1-TARGET_K3,
        "target_k3_fraction":TARGET_K3,
        "target_average_k":TARGET_AVG,
        "tau1":D.TAU1,"tau2":D.TAU2,
        "calibration_rows":len(cal),"cert_rows":len(cert),
        "history":history,
        "certification_routes":cert_routes,
        "cert_pass":passed,
        "allowed_average_k":[CERT_MIN,CERT_MAX],
        "routing_signal":"uncertainty_only",
        "official175_used":False,
        "dev_used_for_threshold_selection":False,
    }
    print("PT57G_BUDGET_CERT:",json.dumps(result,indent=2))
    return result

def set_trainables(model):
    for p in model.parameters(): p.requires_grad=False
    experts=[]
    for n,p in model.named_parameters():
        if n.endswith(".moe_A") or n.endswith(".moe_B"):
            p.requires_grad=True; experts.append(p)
    routers=list(model.moe_router_bank.parameters())
    for p in routers: p.requires_grad=True
    for p in model.moe_criticality_bank.parameters(): p.requires_grad=False
    print("TRAINABLE_EXPERT_PARAMETERS:",sum(p.numel() for p in experts))
    print("TRAINABLE_ROUTER_PARAMETERS:",sum(p.numel() for p in routers))
    return experts,routers

def snapshot(ps):
    return [p.detach().float().cpu().clone() for p in ps]

def delta_l2(before,ps):
    ss=0.0
    for a,p in zip(before,ps):
        d=p.detach().float().cpu()-a
        ss+=float((d*d).sum())
    return math.sqrt(ss)

def save_state(model,outdir,meta):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    state={}
    for n,p in model.named_parameters():
        if n.endswith(".moe_A") or n.endswith(".moe_B") or n.startswith("moe_router_bank."):
            state[n]=p.detach().cpu()
    if len(state)!=624: raise RuntimeError(f"Expected 624 state tensors, got {len(state)}")
    torch.save(state,outdir/"moe_trainable_state.pt")
    (outdir/"moe_config.json").write_text(json.dumps(meta,indent=2))

def run(args):
    train=pd.read_csv(HERE/"data/train281_asr_target_clean_pt57_v2.csv",low_memory=False)
    dev=pd.read_csv(HERE/"data/dev61_asr_target_clean_pt57_v2.csv",low_memory=False)

    if args.smoke:
        # Deterministic spread across the dataset rather than only the first rows.
        train=train.iloc[::max(1,len(train)//12)].head(12).reset_index(drop=True)
        dev=dev.iloc[::max(1,len(dev)//6)].head(6).reset_index(drop=True)
        args.epochs=1; args.patience=1; args.calibration_rows=min(8,len(train)); args.cert_rows=min(4,max(1,len(train)-args.calibration_rows))

    outdir=Path(args.outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"RUN_CONFIG.json").write_text(json.dumps(vars(args),indent=2,default=str))

    print("=== PT57G ASR-SPECIALIZED DYNAMIC-k RECOVERY ===")
    print("foundation: mistralai/Voxtral-Small-24B-2507 (4-bit frozen)")
    print("initialization:",PT15_CKPT)
    print("objective: ASR transcript only")
    print("train rows:",len(train),"dev rows:",len(dev))
    print("expert_lr:",args.expert_lr,"router_lr:",args.router_lr)
    print("epochs:",args.epochs,"patience:",args.patience)
    print("official175_used: False")

    model,processor,device=B.load_base(0)
    D.install_dynamic_moe(model)
    load_pt15_state(model)

    # No row-level criticality used. Heads stay zero/frozen and beta=0.
    D.ALPHA_U=1.0; D.BETA_C=0.0

    budget=exact_budget_calibrate(
        model,processor,train,device,
        cal_rows=args.calibration_rows,cert_rows=args.cert_rows,iters=args.calibration_iters,seed_base=5700
    )
    (outdir/"INITIAL_BUDGET_CERT.json").write_text(json.dumps(budget,indent=2))
    if not budget["cert_pass"]:
        raise RuntimeError("Initial TRAIN-only routing budget certification failed.")

    D.DYNAMIC_ENABLED=True
    pre=evaluate(
        model,processor,dev,device,
        "PT57G PRE-TRAIN FULL DEV" if not args.smoke else "PT57G PRE-TRAIN SMOKE DEV",
        outdir/"dev_pretrain.csv",
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    (outdir/"PRETRAIN_METRICS.json").write_text(json.dumps(pre,indent=2))

    experts,routers=set_trainables(model)
    be=snapshot(experts); br=snapshot(routers)

    opt=AdamW(
        [{"params":experts,"lr":args.expert_lr},{"params":routers,"lr":args.router_lr}],
        weight_decay=args.weight_decay,
    )

    best=float("inf");best_ep=0;best_tie=None;stale=0
    history=[];eg=0;rg=0;t0=time.time()

    for ep in range(1,args.epochs+1):
        D.DYNAMIC_ENABLED=True; D.CURRENT_CRIT_TARGET=None
        D.reset_stats(model); model.train()
        order=train.sample(frac=1.0,random_state=5800+ep).reset_index(drop=True)
        opt.zero_grad(set_to_none=True)
        tasks=[];totals=[];skipped=0
        pb=tqdm(order.iterrows(),total=len(order),desc=f"PT57G ASR ep{ep}",dynamic_ncols=True,mininterval=5)
        for step,(_,r) in enumerate(pb,1):
            batch=B.make_train_batch(processor,r["resolved_audio_path"],B.clean_transcript(r["asr_target_clean"]),device)
            out=model(**batch,use_cache=False,return_dict=True)
            lb,ent=B.moe_aux(model)
            task=out.loss
            loss=task + args.lb_coef*lb
            if not torch.isfinite(loss):
                skipped+=1;opt.zero_grad(set_to_none=True);continue
            (loss/args.grad_accum).backward()
            if any(p.grad is not None and p.grad.detach().abs().sum().item()>0 for p in experts): eg+=1
            if any(p.grad is not None and p.grad.detach().abs().sum().item()>0 for p in routers): rg+=1
            if step%args.grad_accum==0 or step==len(order):
                torch.nn.utils.clip_grad_norm_(experts+routers,args.max_grad_norm)
                opt.step();opt.zero_grad(set_to_none=True)
            tasks.append(float(task.detach()));totals.append(float(loss.detach()))
            pb.set_postfix(task=f"{tasks[-1]:.3f}",lb=f"{float(lb.detach()):.3f}",ent=f"{float(ent.detach()):.3f}")
            del batch,out,task,loss,lb,ent
        pb.close();gc.collect();torch.cuda.empty_cache()

        # Recalibrate only on TRAIN because router/expert changes alter risk states.
        budget_ep=exact_budget_calibrate(
            model,processor,train,device,
            cal_rows=args.calibration_rows,cert_rows=args.cert_rows,iters=max(1,args.post_epoch_calibration_iters),
            seed_base=5700+ep*10
        )
        (outdir/f"BUDGET_CERT_EPOCH_{ep}.json").write_text(json.dumps(budget_ep,indent=2))
        D.DYNAMIC_ENABLED=True
        metrics=evaluate(
            model,processor,dev,device,f"PT57G DEV EPOCH {ep}",outdir/f"dev_epoch_{ep}.csv",
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        rec={
            "epoch":ep,
            "train_task_loss":sum(tasks)/max(1,len(tasks)),
            "train_total_loss":sum(totals)/max(1,len(totals)),
            "skipped":skipped,
            "budget_cert_pass":budget_ep["cert_pass"],
            "train_cert_avg_k":budget_ep["certification_routes"]["average_routed_k"],
            "tau1":D.TAU1,"tau2":D.TAU2,
            **metrics,
        }
        history.append(rec)
        pd.DataFrame(history).to_csv(outdir/"HISTORY.csv",index=False)

        # Selection: WER, then corpus WER, then blanks, then -TokenF1.
        tie=(metrics["corpus_WER"],metrics["blank_predictions"],-metrics["mean_TokenF1"])
        valid=budget_ep["cert_pass"] and metrics["average_routed_k"]<2.0
        improved=valid and (
            metrics["mean_row_WER_Ehsan_style"] < best-1e-12
            or (abs(metrics["mean_row_WER_Ehsan_style"]-best)<=1e-12 and (best_tie is None or tie<best_tie))
        )
        if improved:
            best=metrics["mean_row_WER_Ehsan_style"];best_tie=tie;best_ep=ep;stale=0
            meta={
                "foundation":"mistralai/Voxtral-Small-24B-2507",
                "initialization":"PT15 GLOBAL_BEST Dynamic-k Scratch-MoE",
                "architecture":"4-expert internal Scratch-MoE; uncertainty-only Dynamic-k",
                "objective":"ASR transcript only",
                "experts":4,"rank":16,"alpha":32,"dropout":0.05,
                "expert_lr":args.expert_lr,"router_lr":args.router_lr,
                "best_epoch":ep,"best_metrics":metrics,
                "tau1":D.TAU1,"tau2":D.TAU2,
                "budget_cert":budget_ep,
                "official175_used":False,
                "model_selection_set":"DEV61 only",
            }
            save_state(model,outdir/"GLOBAL_BEST",meta)
            shutil.copy2(outdir/f"dev_epoch_{ep}.csv",outdir/"GLOBAL_BEST/dev_predictions.csv")
            print(f"*** NEW PT57G BEST WER={best:.6f} avgk={metrics['average_routed_k']:.4f} ep={ep} ***")
        else:
            stale+=1
            print(f"no valid DEV improvement {stale}/{args.patience}; best={best:.6f}")
        if stale>=args.patience:
            print("EARLY STOP at epoch",ep)
            break

    cert={
        "pretrain_metrics":pre,
        "best_epoch":best_ep,
        "best_mean_row_WER":None if best==float("inf") else best,
        "expert_gradient_nonzero_steps":eg,
        "router_gradient_nonzero_steps":rg,
        "expert_delta_l2":delta_l2(be,experts),
        "router_delta_l2":delta_l2(br,routers),
        "train_runtime_sec":time.time()-t0,
        "official175_used":False,
        "frozen_dev_targets_after_selection_started":True,
    }
    (outdir/"PT57G_TRAINING_CERT.json").write_text(json.dumps(cert,indent=2))
    print("PT57G_TRAINING_CERT:",json.dumps(cert,indent=2))
    print("DONE:",outdir)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--patience",type=int,default=2)
    ap.add_argument("--expert-lr",type=float,default=5e-5)
    ap.add_argument("--router-lr",type=float,default=1e-5)
    ap.add_argument("--grad-accum",type=int,default=4)
    ap.add_argument("--weight-decay",type=float,default=0.01)
    ap.add_argument("--max-grad-norm",type=float,default=1.0)
    ap.add_argument("--lb-coef",type=float,default=0.005)
    ap.add_argument("--calibration-rows",type=int,default=64)
    ap.add_argument("--cert-rows",type=int,default=24)
    ap.add_argument("--calibration-iters",type=int,default=2)
    ap.add_argument("--post-epoch-calibration-iters",type=int,default=1)
    # Keep the historical decoder fixed during model selection. Sweep comes next.
    ap.add_argument("--max-new-tokens",type=int,default=256)
    ap.add_argument("--repetition-penalty",type=float,default=1.3)
    ap.add_argument("--no-repeat-ngram-size",type=int,default=3)
    args=ap.parse_args()
    run(args)

if __name__=="__main__":
    main()
