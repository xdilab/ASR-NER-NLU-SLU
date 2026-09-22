#!/usr/bin/env python3
from __future__ import annotations

import argparse, importlib.util, json, math, os, random, time, weakref
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# Import the already-certified V1.2 architecture helpers.
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("fixed_top2_base",HERE/"fixed_top2_base.py")
B=importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

FIXED_CKPT=Path("/data/smgreen1/voxtral_moe_v1/internal_scratch_moe_fixed_top2_v1_2/GLOBAL_BEST/moe_trainable_state.pt")

N_EXPERTS=4
MAX_K=3
ALPHA_U=0.55
BETA_C=0.45

# Calibration target: 30% k=1, 55% k=2, 15% k=3 => mean k ≈ 1.85.
Q_K1=0.30
Q_K3_START=0.85

PT108_ADAPTER_BANK=None  # original path when disabled
DYNAMIC_ENABLED=False
TAU1=0.0
TAU2=1.0
CURRENT_CRIT_TARGET=None
COLLECT_RISK=False
RISK_SAMPLES=[]

# PT15 exact-budget calibration support.
# The original collector kept only up to 32 strided risk values per routing
# forward, which was useful diagnostically but biased for quantile calibration.
# PT15 can instead accumulate a GPU histogram over EVERY routed-token risk.
COLLECT_RISK_HIST=False
RISK_HIST=None
RISK_HIST_BINS=8192

# Safety/operational criticality supervision derived ONLY from TRAIN annotations.
# Max severity among present fields becomes the row-level weak target.
# official175 is never used.
CRITICALITY_SEVERITY={
    "ann__slot__runway_number":1.0,
    "ann__slot__approach_runway":1.0,
    "ann__slot__target_altitude":1.0,
    "ann__slot__flight_level":1.0,
    "ann__slot__target_heading":1.0,
    "ann__slot__transfer_frequency":1.0,
    "ann__slot__target_speed":0.9,
    "ann__slot__taxiway":0.9,
    "ann__slot__holding_point":0.9,
    "ann__slot__clearance_limit":0.9,
    "ann__slot__full_callsign":0.8,
    "ann__slot__callsign_code":0.8,
    "ann__slot__waypoint_fix":0.8,
    "ann__slot__planned_route":0.8,
    "ann__slot__departure_route":0.8,
    "ann__slot__arrival_route":0.8,
    "ann__slot__surface_route":0.8,
    "ann__slot__contact_unit":0.7,
}

def present(v):
    if pd.isna(v): return False
    s=str(v).strip().lower()
    return s not in ("","nan","none","n/a","na")

def row_criticality(row):
    scores=[
        sev for col,sev in CRITICALITY_SEVERITY.items()
        if col in row.index and present(row[col])
    ]
    return float(max(scores) if scores else 0.0)

class CriticalityBank(nn.ModuleDict):
    pass

class RiskAdaptiveMoELoraLinear(B.SparseMoELoraLinear):
    """
    Same four Scratch-MoE LoRA experts as Fixed Top-2, plus a lightweight
    criticality head sharing the same local token state.

    Expert router answers: WHICH experts?
    Risk controller answers: HOW MANY experts? (k=1/2/3)

    risk = 0.55 * normalized_router_entropy + 0.45 * predicted_criticality
    """
    def __init__(self,*args,criticality_bank=None,**kwargs):
        super().__init__(*args,**kwargs)
        object.__setattr__(self,"_criticality_bank_ref",weakref.ref(criticality_bank))
        self.last_crit_loss=None
        self.register_buffer("k_counts",torch.zeros(4,dtype=torch.long,device=self.route_counts.device),persistent=False)
        self.register_buffer("risk_sum",torch.zeros((),dtype=torch.float32,device=self.route_counts.device),persistent=False)
        self.register_buffer("crit_sum",torch.zeros((),dtype=torch.float32,device=self.route_counts.device),persistent=False)
        self.register_buffer("risk_token_count",torch.zeros((),dtype=torch.long,device=self.route_counts.device),persistent=False)

    def crit_head(self):
        b=self._criticality_bank_ref()
        if b is None:
            raise RuntimeError("Criticality bank reference lost.")
        return b[self.router_key]

    def reset_stats(self):
        super().reset_stats()
        self.k_counts.zero_()
        self.risk_sum.zero_()
        self.crit_sum.zero_()
        self.risk_token_count.zero_()
        self.last_crit_loss=None

    def forward(self,x,*args,**kwargs):
        global DYNAMIC_ENABLED,TAU1,TAU2,CURRENT_CRIT_TARGET,COLLECT_RISK,RISK_SAMPLES

        base_y=self.base_layer(x,*args,**kwargs)
        shape=x.shape
        xf=x.reshape(-1,shape[-1])
        if xf.numel()==0:
            return base_y

        router=self.router()
        if xf.shape[-1] != router.in_features:
            raise RuntimeError(
                f"Dynamic-k router shape mismatch router={self.router_key} "
                f"x={xf.shape[-1]} expected={router.in_features}"
            )

        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=xf.is_cuda):
            original_logits=router(xf)
            logits=original_logits
            if PT108_ADAPTER_BANK is not None:
                logits=original_logits + PT108_ADAPTER_BANK[self.router_key](xf).to(original_logits.dtype)
            clogit=self.crit_head()(xf).squeeze(-1)

        probs=torch.softmax(logits.float(),dim=-1)
        crit=torch.sigmoid(clogit.float())

        # Per-token router uncertainty.
        ent=-(probs*torch.log(probs.clamp_min(1e-9))).sum(dim=-1)/math.log(self.n_experts)
        risk=ALPHA_U*ent + BETA_C*crit

        # Weakly supervised criticality head on the SAME internal token states
        # used at inference, avoiding the old pooled-feature mismatch.
        if self.record_aux and CURRENT_CRIT_TARGET is not None:
            tgt=torch.full_like(crit,float(CURRENT_CRIT_TARGET))
            self.last_crit_loss=F.mse_loss(crit,tgt)

        if self.record_aux and COLLECT_RISK:
            # Legacy diagnostic subsample.
            rr=risk.detach().flatten()
            stride=max(1,int(math.ceil(rr.numel()/32)))
            RISK_SAMPLES.extend(rr[::stride][:32].float().cpu().tolist())

        # PT15: unbiased all-token histogram used for budget calibration.
        # risk is bounded in [0,1] for the supported uncertainty/criticality
        # mixtures. Keep the histogram on GPU to avoid per-forward CPU sync.
        global COLLECT_RISK_HIST, RISK_HIST, RISK_HIST_BINS
        if self.record_aux and COLLECT_RISK_HIST:
            rr=risk.detach().float().flatten().clamp_(0.0,1.0)
            h=torch.histc(rr,bins=RISK_HIST_BINS,min=0.0,max=1.0)
            if RISK_HIST is None:
                RISK_HIST=h
            else:
                RISK_HIST.add_(h)

        if DYNAMIC_ENABLED:
            k=torch.full((xf.shape[0],),2,device=xf.device,dtype=torch.long)
            k[risk < TAU1]=1
            k[risk >= TAU2]=3
        else:
            k=torch.full((xf.shape[0],),2,device=xf.device,dtype=torch.long)

        topv,topi=torch.topk(probs,k=MAX_K,dim=-1)
        ranks=torch.arange(MAX_K,device=xf.device).view(1,-1)
        active_rank=(ranks < k.unsqueeze(-1))
        selected_v=topv*active_rank.float()
        gates=selected_v/selected_v.sum(dim=-1,keepdim=True).clamp_min(1e-8)

        xdrop=F.dropout(xf,p=self.dropout_p,training=self.training)
        delta=torch.zeros(xf.shape[0],self.out_features,device=base_y.device,dtype=base_y.dtype)

        for e in range(self.n_experts):
            emask=(topi==e) & active_rank
            active=emask.any(dim=-1)
            if not bool(active.any()):
                continue
            w=(gates[active]*emask[active].float()).sum(dim=-1)
            xe=xdrop[active]
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=xe.is_cuda):
                low=F.linear(xe,self.moe_A[e])
                up=F.linear(low,self.moe_B[e])*self.scale
            delta[active]+=up.to(delta.dtype)*w.to(delta.dtype).unsqueeze(-1)

        if self.record_aux:
            if PT108_ADAPTER_BANK is not None:
                reference_probability=torch.softmax(original_logits.detach().float(),dim=-1)
                self.last_pt108_kl_loss=F.kl_div(torch.log_softmax(logits.float(),dim=-1),reference_probability,reduction='batchmean')
            else:
                self.last_pt108_kl_loss=None
            mean_prob=probs.mean(dim=0)
            # Load statistic uses the actually active routes.
            hard=F.one_hot(topi,num_classes=self.n_experts).float()
            hard=hard*active_rank.unsqueeze(-1).float()
            active_per_token=k.float().clamp_min(1.0)
            mean_load=(hard.sum(dim=1)/active_per_token.unsqueeze(-1)).mean(dim=0).detach()
            self.last_lb_loss=self.n_experts*torch.sum(mean_prob*mean_load)
            self.last_entropy=ent.mean()

            with torch.no_grad():
                selected_oh=F.one_hot(topi,num_classes=self.n_experts).long()
                selected_oh=selected_oh*active_rank.unsqueeze(-1).long()
                self.route_counts += selected_oh.sum(dim=(0,1)).to(self.route_counts.dtype)
                self.entropy_sum += ent.detach().sum().to(self.entropy_sum.dtype)
                self.routed_token_count += int(ent.numel())
                for kval in (1,2,3):
                    self.k_counts[kval] += int((k==kval).sum())
                self.risk_sum += risk.detach().sum().to(self.risk_sum.dtype)
                self.crit_sum += crit.detach().sum().to(self.crit_sum.dtype)
                self.risk_token_count += int(risk.numel())

        return base_y+delta.reshape(*base_y.shape)

def install_dynamic_moe(model):
    # Freeze entire pretrained foundation.
    for p in model.parameters():
        p.requires_grad=False

    candidates=[]
    for name,module in list(model.named_modules()):
        leaf=name.rsplit(".",1)[-1]
        if leaf in B.TARGETS and hasattr(module,"in_features") and hasattr(module,"out_features"):
            candidates.append((name,module))
    if not candidates:
        raise RuntimeError("No q/k/v/o projections found.")

    routers=B.RouterBank()
    crits=CriticalityBank()
    model.moe_router_bank=routers
    model.moe_criticality_bank=crits

    group_to_key={}
    members=defaultdict(list)
    for name,module in candidates:
        prefix=name.rsplit(".",1)[0]
        dim=int(module.in_features)
        group=(prefix,dim)
        members[group].append(name)
        if group not in group_to_key:
            key=f"router_{len(group_to_key):03d}"
            group_to_key[group]=key
            dev=next(module.parameters()).device

            r=nn.Linear(dim,N_EXPERTS,bias=False,device=dev,dtype=torch.float32)
            nn.init.normal_(r.weight,mean=0.0,std=1e-3)
            routers[key]=r

            c=nn.Linear(dim,1,bias=True,device=dev,dtype=torch.float32)
            nn.init.zeros_(c.weight)
            nn.init.zeros_(c.bias)
            crits[key]=c

    aux_owner={}
    for group,names in members.items():
        q=[n for n in names if n.endswith(".q_proj")]
        aux_owner[group]=q[0] if q else names[0]

    for name,module in candidates:
        parent,attr=B._get_parent(model,name)
        prefix=name.rsplit(".",1)[0]
        group=(prefix,int(module.in_features))
        w=RiskAdaptiveMoELoraLinear(
            module,routers,group_to_key[group],
            n_experts=N_EXPERTS,top_k=2,rank=B.RANK,
            alpha=B.LORA_ALPHA,dropout=B.LORA_DROPOUT,
            record_aux=(name==aux_owner[group]),
            criticality_bank=crits,
        )
        setattr(parent,attr,w)

    print("DYNAMIC_K_ROUTER_GROUPS:",len(group_to_key))
    print("DYNAMIC_K_CRITICALITY_HEADS:",len(crits))
    return group_to_key

def load_fixed_state(model):
    if not FIXED_CKPT.exists():
        raise FileNotFoundError(f"Fixed Top-2 checkpoint missing: {FIXED_CKPT}")
    state=torch.load(FIXED_CKPT,map_location="cpu")
    named=dict(model.named_parameters())
    loaded=[];missing=[];shape_bad=[]
    for n,t in state.items():
        if n not in named:
            missing.append(n); continue
        if tuple(named[n].shape)!=tuple(t.shape):
            shape_bad.append((n,tuple(named[n].shape),tuple(t.shape))); continue
        named[n].data.copy_(t.to(device=named[n].device,dtype=named[n].dtype))
        loaded.append(n)
    print("FIXED_TOP2_STATE_TENSORS:",len(state))
    print("FIXED_TOP2_LOADED_TENSORS:",len(loaded))
    print("FIXED_TOP2_UNMATCHED:",len(missing))
    print("FIXED_TOP2_SHAPE_BAD:",len(shape_bad))
    if missing or shape_bad or len(loaded)!=len(state):
        print("missing preview:",missing[:10])
        print("shape_bad preview:",shape_bad[:10])
        raise RuntimeError("Fixed Top-2 state did not load exactly.")
    return len(loaded)

def wrappers(model):
    return [m for m in model.modules() if isinstance(m,RiskAdaptiveMoELoraLinear) and m.record_aux]

def reset_stats(model):
    for m in wrappers(model): m.reset_stats()

def crit_loss(model):
    ls=[m.last_crit_loss for m in wrappers(model) if m.last_crit_loss is not None]
    if not ls:
        p=next(model.moe_criticality_bank.parameters())
        return p.sum()*0.0
    return torch.stack(ls).mean()

def route_summary(model):
    counts=torch.zeros(N_EXPERTS,dtype=torch.long)
    kcounts=torch.zeros(4,dtype=torch.long)
    es=0.0; rs=0.0; cs=0.0; tok=0
    for m in wrappers(model):
        counts+=m.route_counts.detach().cpu()
        kcounts+=m.k_counts.detach().cpu()
        es+=float(m.entropy_sum.detach().cpu())
        rs+=float(m.risk_sum.detach().cpu())
        cs+=float(m.crit_sum.detach().cpu())
        tok+=int(m.risk_token_count.detach().cpu())
    selections=max(1,int(counts.sum()))
    token_total=max(1,int(kcounts[1:].sum()))
    avgk=sum(k*int(kcounts[k]) for k in (1,2,3))/token_total
    return {
        "expert_selected_counts":[int(x) for x in counts],
        "expert_selection_share":[float(x/selections) for x in counts],
        "k1_fraction":float(kcounts[1]/token_total),
        "k2_fraction":float(kcounts[2]/token_total),
        "k3_fraction":float(kcounts[3]/token_total),
        "average_routed_k":float(avgk),
        "expert_activation_reduction_vs_fixed_top2":float((2.0-avgk)/2.0),
        "normalized_router_entropy":float(es/max(1,tok)),
        "mean_predicted_criticality":float(cs/max(1,tok)),
        "mean_risk":float(rs/max(1,tok)),
        "all_experts_used":bool((counts>0).all()),
    }

def controller_delta(initial,model):
    sq=0.0
    named=dict(model.moe_criticality_bank.named_parameters())
    for n,p in named.items():
        d=p.detach().float().cpu()-initial[n]
        sq+=float((d*d).sum())
    return math.sqrt(sq)

def train_controller(model,processor,train_df,device,epochs=1,grad_accum=4,lr=2e-4,smoke=False):
    global CURRENT_CRIT_TARGET,DYNAMIC_ENABLED
    DYNAMIC_ENABLED=False

    # Freeze fixed expert/router state; train ONLY new operational-criticality heads.
    for p in model.parameters(): p.requires_grad=False
    for p in model.moe_criticality_bank.parameters(): p.requires_grad=True

    params=list(model.moe_criticality_bank.parameters())
    initial={n:p.detach().float().cpu().clone() for n,p in model.moe_criticality_bank.named_parameters()}
    opt=AdamW(params,lr=lr,weight_decay=0.01)

    history=[]
    for ep in range(1,epochs+1):
        order=train_df.sample(frac=1.0,random_state=800+ep).reset_index(drop=True)
        opt.zero_grad(set_to_none=True)
        losses=[]; grad_steps=0
        pb=tqdm(order.iterrows(),total=len(order),desc=f"CRITICALITY-CONTROLLER ep{ep}",dynamic_ncols=True,mininterval=5)
        for step,(_,r) in enumerate(pb,1):
            CURRENT_CRIT_TARGET=row_criticality(r)
            b=B.make_train_batch(processor,r["resolved_audio_path"],B.clean_transcript(r["asr_target_clean"]),device)
            model.train()
            _=model(**b,use_cache=False,return_dict=True)
            loss=crit_loss(model)
            (loss/grad_accum).backward()
            if any(p.grad is not None and p.grad.detach().abs().sum().item()>0 for p in params):
                grad_steps+=1
            if step%grad_accum==0 or step==len(order):
                torch.nn.utils.clip_grad_norm_(params,1.0)
                opt.step();opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            pb.set_postfix(mse=f"{losses[-1]:.4f}",target=f"{CURRENT_CRIT_TARGET:.1f}")
        pb.close()
        history.append({
            "epoch":ep,
            "mean_criticality_mse":sum(losses)/max(1,len(losses)),
            "gradient_nonzero_steps":grad_steps,
        })
    CURRENT_CRIT_TARGET=None
    cert={
        "history":history,
        "criticality_head_delta_l2":controller_delta(initial,model),
        "train_label_mean":float(train_df.apply(row_criticality,axis=1).mean()),
    }
    print("CONTROLLER_CERT:",json.dumps(cert,indent=2))
    return cert

@torch.no_grad()
def calibrate(model,processor,df,device,n=40):
    global COLLECT_RISK,RISK_SAMPLES,CURRENT_CRIT_TARGET,DYNAMIC_ENABLED,TAU1,TAU2
    DYNAMIC_ENABLED=False
    RISK_SAMPLES=[]
    COLLECT_RISK=True
    sample=df.head(min(n,len(df)))
    pb=tqdm(sample.iterrows(),total=len(sample),desc="RISK-CALIBRATE",dynamic_ncols=True,mininterval=5)
    for _,r in pb:
        CURRENT_CRIT_TARGET=None
        b=B.make_train_batch(processor,r["resolved_audio_path"],B.clean_transcript(r["asr_target_clean"]),device)
        model.eval()
        _=model(**b,use_cache=False,return_dict=True)
    pb.close()
    COLLECT_RISK=False
    CURRENT_CRIT_TARGET=None
    if len(RISK_SAMPLES)<100:
        raise RuntimeError("Too few risk samples for calibration.")
    vals=torch.tensor(RISK_SAMPLES,dtype=torch.float32)
    TAU1=float(torch.quantile(vals,Q_K1))
    TAU2=float(torch.quantile(vals,Q_K3_START))
    if not TAU1<TAU2:
        raise RuntimeError(f"Invalid thresholds: {TAU1}, {TAU2}")
    out={
        "n_risk_samples":len(RISK_SAMPLES),
        "tau1":TAU1,
        "tau2":TAU2,
        "risk_mean":float(vals.mean()),
        "risk_std":float(vals.std()),
        "target_k1_fraction":Q_K1,
        "target_k2_fraction":Q_K3_START-Q_K1,
        "target_k3_fraction":1.0-Q_K3_START,
        "target_average_k":1*Q_K1+2*(Q_K3_START-Q_K1)+3*(1-Q_K3_START),
    }
    print("CALIBRATION:",json.dumps(out,indent=2))
    return out

@torch.no_grad()
def evaluate_dynamic(model,processor,df,device,label,out_csv):
    global DYNAMIC_ENABLED
    DYNAMIC_ENABLED=True
    reset_stats(model)
    rows=[];refs=[];hyps=[]
    print(f"\n=== {label} ===")
    for i,(_,r) in enumerate(df.iterrows(),1):
        g=B.clean_transcript(r["asr_target_clean"])
        p,sec=B.infer_one(model,processor,r["resolved_audio_path"],device)
        rw=B.row_wer(g,p)
        refs.append(g);hyps.append(p)
        rows.append({
            "audio_name":r["audio_name"],"reference_transcript":g,
            "predicted_transcript":p,"row_wer_ehsan":rw,
            "row_cer":B.row_cer(g,p),"token_f1":B.token_f1(g,p),
            "runtime_sec":sec,"blank":int(not p),
            "row_operational_criticality":row_criticality(r),
        })
        if i<=5 or i%25==0 or i==len(df):
            print(f"[{i:04d}/{len(df)}] rowWER={rw:.4f} mean={sum(x['row_wer_ehsan'] for x in rows)/len(rows):.4f}")
    d=pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(out_csv,index=False)
    m={
        "n_rows":len(d),
        "mean_row_WER_Ehsan_style":float(d.row_wer_ehsan.mean()),
        "mean_row_CER":float(d.row_cer.mean()),
        "corpus_WER":float(B.corpus_wer(refs,hyps)),
        "mean_TokenF1":float(d.token_f1.mean()),
        "mean_runtime_sec":float(d.runtime_sec.mean()),
        "blank_predictions":int(d.blank.sum()),
        **route_summary(model),
    }
    print("DYNAMIC_K_METRICS:",json.dumps(m,indent=2))
    return m

def save_controller(model,outdir,calib,cert,metrics):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    torch.save(
        {n:p.detach().cpu() for n,p in model.moe_criticality_bank.named_parameters()},
        outdir/"criticality_controller_state.pt"
    )
    summary={
        "architecture":"Risk-Adaptive Dynamic-k over certified Scratch-MoE",
        "fixed_top2_checkpoint":str(FIXED_CKPT),
        "experts":4,
        "k_values":[1,2,3],
        "alpha_uncertainty":ALPHA_U,
        "beta_criticality":BETA_C,
        "criticality_supervision":"train-only ATC slot-derived weak operational criticality",
        "criticality_granularity":"row-level weak target learned at same internal token states",
        "calibration":calib,
        "controller_cert":cert,
        "dev_metrics":metrics,
        "official175_used":False,
        "status":"initial dynamic-k controller result; token/entity-level criticality refinement remains future work",
    }
    (outdir/"DYNAMIC_K_SUMMARY.json").write_text(json.dumps(summary,indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split-dir",default=str(HERE/"data"))
    ap.add_argument("--outdir",default="/data/smgreen1/voxtral_moe_v1/risk_adaptive_dynamic_k_v1_0")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--controller-epochs",type=int,default=1)
    ap.add_argument("--controller-lr",type=float,default=2e-4)
    args=ap.parse_args()

    train_df,dev_df=B.validate_data(Path(args.split_dir))
    if args.smoke:
        train_df=train_df.head(12).copy()
        dev_df=dev_df.head(6).copy()
        args.outdir="/data/smgreen1/voxtral_moe_v1/risk_adaptive_dynamic_k_SMOKE_v1_0"

    print("=== RISK-ADAPTIVE DYNAMIC-k ATC MoE ===")
    print("train:",len(train_df),"dev:",len(dev_df),"official175_used: False")
    print("fixed_top2_reference_WER: 0.32330218758554546")
    print("strong_qlora_reference_WER: 0.3154370526978759")
    print("risk = 0.55*router_uncertainty + 0.45*learned_ATC_criticality")
    print("k choices: 1 / 2 / 3")
    print("shared_expert: False")

    model,processor,device=B.load_base(0)
    install_dynamic_moe(model)
    load_fixed_state(model)

    cert=train_controller(
        model,processor,train_df,device,
        epochs=args.controller_epochs,
        grad_accum=2 if args.smoke else 4,
        lr=args.controller_lr,
        smoke=args.smoke
    )

    calib=calibrate(model,processor,train_df,device,n=6 if args.smoke else 40)

    metrics=evaluate_dynamic(
        model,processor,dev_df,device,
        "RISK-ADAPTIVE DYNAMIC-k DEV",
        Path(args.outdir)/"dynamic_k_dev_predictions.csv"
    )

    save_controller(model,args.outdir,calib,cert,metrics)
    print("DONE:",args.outdir)

if __name__=="__main__":
    main()
