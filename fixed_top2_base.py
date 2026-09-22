#!/usr/bin/env python3
from __future__ import annotations

import argparse, gc, json, math, os, random, re, time, weakref
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, VoxtralForConditionalGeneration

REPO_ID = "mistralai/Voxtral-Small-24B-2507"
PROMPT = "You are an aviation expert now transcribe this audio only provide the text transcribe nothing more than that."

N_EXPERTS = 4
TOP_K = 2
RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
TARGETS = {"q_proj","k_proj","v_proj","o_proj"}

# This is deliberately NO-SHARED-EXPERT.
# Voxtral is pretrained; all MoE routers and LoRA experts are newly initialized.

def clean(x):
    return re.sub(r"\s+"," ","" if x is None else str(x)).strip()

def clean_transcript(x):
    s="" if x is None else str(x)
    s=re.sub(r"\[[^\]]+\]"," ",s)
    return clean(s)

def lev(a,b):
    prev=list(range(len(b)+1))
    for i,ai in enumerate(a,1):
        cur=[i]+[0]*len(b)
        for j,bj in enumerate(b,1):
            cur[j]=min(cur[j-1]+1,prev[j]+1,prev[j-1]+(ai!=bj))
        prev=cur
    return prev[-1]

def row_wer(g,p):
    gw,pw=g.split(),p.split()
    return lev(gw,pw)/max(1,len(gw))

def row_cer(g,p):
    return lev(list(g),list(p))/max(1,len(g))

def corpus_wer(refs,hyps):
    e=n=0
    for g,p in zip(refs,hyps):
        gw,pw=g.split(),p.split()
        e += lev(gw,pw); n += len(gw)
    return e/max(1,n)

def token_f1(g,p):
    gc,pc=Counter(g.split()),Counter(p.split())
    if not gc and not pc: return 1.0
    if not gc or not pc: return 0.0
    ov=sum((gc&pc).values())
    pr=ov/sum(pc.values()); rc=ov/sum(gc.values())
    return 0.0 if pr+rc==0 else 2*pr*rc/(pr+rc)

def build_conv(audio_path):
    return [{"role":"user","content":[
        {"type":"audio","path":str(audio_path)},
        {"type":"text","text":PROMPT},
    ]}]

def prompt_inputs(processor,audio_path,device):
    x=processor.apply_chat_template(build_conv(audio_path))
    return x.to(device=device)

def make_train_batch(processor,audio_path,target,device):
    p=prompt_inputs(processor,audio_path,device)
    tid=processor.tokenizer(
        target,return_tensors="pt",add_special_tokens=False,
        truncation=True,max_length=448
    ).input_ids.to(device)
    eos=processor.tokenizer.eos_token_id
    if eos is not None:
        tid=torch.cat([tid,torch.tensor([[int(eos)]],device=device,dtype=tid.dtype)],dim=1)
    ids=torch.cat([p["input_ids"],tid],dim=1)
    mask=torch.cat([p["attention_mask"],torch.ones_like(tid)],dim=1)
    labels=torch.cat([torch.full_like(p["input_ids"],-100),tid],dim=1)
    extra={k:v for k,v in p.items() if k not in ("input_ids","attention_mask")}
    return dict(input_ids=ids,attention_mask=mask,labels=labels,**extra)

def find_local_voxtral_snapshot():
    repo_dir="models--mistralai--Voxtral-Small-24B-2507"
    roots=[]
    for raw in [
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        str(Path(os.environ["HF_HOME"])/"hub") if os.environ.get("HF_HOME") else None,
        "/data/smgreen1/hf_cache/hub",
        str(Path.home()/".cache/huggingface/hub"),
    ]:
        if raw and raw not in roots: roots.append(raw)
    for root in roots:
        sr=Path(root)/repo_dir/"snapshots"
        if not sr.exists(): continue
        for p in sr.iterdir():
            if not p.is_dir(): continue
            shards=list(p.glob("model-*-of-*.safetensors"))
            if (p/"model.safetensors.index.json").exists() and len(shards)>=11:
                print("VOXTRAL_LOCAL_SNAPSHOT:",p)
                print("VOXTRAL_LOCAL_SHARDS:",len(shards))
                return str(p)
    return None

def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def load_base(device=0):
    snap=find_local_voxtral_snapshot()
    source=snap or REPO_ID
    local=bool(snap)
    print("MODEL_SOURCE:",source)
    model=VoxtralForConditionalGeneration.from_pretrained(
        source,
        quantization_config=bnb_config(),
        device_map={"":device},
        low_cpu_mem_usage=True,
        local_files_only=local,
    )
    processor=AutoProcessor.from_pretrained(source if snap else REPO_ID,local_files_only=local)
    model.config.use_cache=False

    # Critical for custom trainables under gradient checkpointing.
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant":False}
        )
        print("GRADIENT_CHECKPOINTING: use_reentrant=False")
    except TypeError:
        model.gradient_checkpointing_enable()
        print("GRADIENT_CHECKPOINTING: default API")
    return model,processor,next(model.parameters()).device

class RouterBank(nn.ModuleDict):
    pass

class SparseMoELoraLinear(nn.Module):
    """
    Frozen base projection + 4 newly initialized LoRA experts.
    A router shared by q/k/v/o within the same attention layer chooses Top-2.
    No shared expert.
    """
    def __init__(
        self, base_layer, router_bank, router_key,
        n_experts=4, top_k=2, rank=16, alpha=32.0, dropout=0.05,
        record_aux=False
    ):
        super().__init__()
        self.base_layer=base_layer
        for p in self.base_layer.parameters():
            p.requires_grad=False

        self.in_features=int(base_layer.in_features)
        self.out_features=int(base_layer.out_features)
        self.n_experts=int(n_experts)
        self.top_k=int(top_k)
        self.rank=int(rank)
        self.scale=float(alpha)/float(rank)
        self.dropout_p=float(dropout)
        self.router_key=str(router_key)
        self.record_aux=bool(record_aux)

        # Avoid registering RouterBank again inside every projection wrapper.
        object.__setattr__(self,"_router_bank_ref",weakref.ref(router_bank))

        dev = next(base_layer.parameters()).device
        # Keep trainable parameters in fp32; autocast makes low-rank matmuls BF16.
        self.moe_A=nn.Parameter(torch.empty(self.n_experts,self.rank,self.in_features,device=dev,dtype=torch.float32))
        self.moe_B=nn.Parameter(torch.empty(self.n_experts,self.out_features,self.rank,device=dev,dtype=torch.float32))

        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(self.moe_A[e],a=math.sqrt(5))
            # Tiny non-zero B gives the router a learning signal immediately while
            # keeping the initial delta extremely close to zero.
            nn.init.normal_(self.moe_B[e],mean=0.0,std=1e-4)

        self.last_lb_loss=None
        self.last_entropy=None
        self.register_buffer("route_counts",torch.zeros(self.n_experts,dtype=torch.long,device=dev),persistent=False)
        self.register_buffer("entropy_sum",torch.zeros((),dtype=torch.float32,device=dev),persistent=False)
        self.register_buffer("routed_token_count",torch.zeros((),dtype=torch.long,device=dev),persistent=False)

    def router(self):
        bank=self._router_bank_ref()
        if bank is None:
            raise RuntimeError("Router bank reference was lost.")
        return bank[self.router_key]

    def reset_stats(self):
        self.route_counts.zero_()
        self.entropy_sum.zero_()
        self.routed_token_count.zero_()

    def forward(self,x,*args,**kwargs):
        base_y=self.base_layer(x,*args,**kwargs)
        shape=x.shape
        xf=x.reshape(-1,shape[-1])
        if xf.numel()==0:
            return base_y

        # Router params remain fp32, but the matmul is BF16 under autocast.
        router=self.router()
        if xf.shape[-1] != router.in_features:
            raise RuntimeError(
                f"Scratch-MoE router shape mismatch after shape-aware install: "
                f"router_key={self.router_key} x_dim={xf.shape[-1]} "
                f"router_in={router.in_features} base_in={self.in_features}"
            )
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=xf.is_cuda):
            logits=router(xf)
        probs=torch.softmax(logits.float(),dim=-1)
        topv,topi=torch.topk(probs,k=self.top_k,dim=-1)
        gates=topv/topv.sum(dim=-1,keepdim=True).clamp_min(1e-8)

        xdrop=F.dropout(xf,p=self.dropout_p,training=self.training)
        delta=torch.zeros(
            xf.shape[0],self.out_features,
            device=base_y.device,dtype=base_y.dtype
        )

        # Compute ONLY the selected expert paths.
        for e in range(self.n_experts):
            mask=(topi==e)
            active=mask.any(dim=-1)
            if not bool(active.any()):
                continue
            w=(gates[active]*mask[active].float()).sum(dim=-1)
            xe=xdrop[active]
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=xe.is_cuda):
                low=F.linear(xe,self.moe_A[e])
                up=F.linear(low,self.moe_B[e]) * self.scale
            delta[active] += up.to(delta.dtype)*w.to(delta.dtype).unsqueeze(-1)

        if self.record_aux:
            # Switch-style differentiable load balance + normalized entropy.
            mean_prob=probs.mean(dim=0)
            hard=F.one_hot(topi,num_classes=self.n_experts).float().sum(dim=1)/float(self.top_k)
            mean_load=hard.mean(dim=0).detach()
            self.last_lb_loss=self.n_experts*torch.sum(mean_prob*mean_load)
            ent=-(probs*torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
            ent=ent/math.log(self.n_experts)
            self.last_entropy=ent.mean()

            with torch.no_grad():
                self.route_counts += F.one_hot(topi,num_classes=self.n_experts).sum(dim=(0,1)).to(self.route_counts.dtype)
                self.entropy_sum += ent.detach().sum().to(self.entropy_sum.dtype)
                self.routed_token_count += int(ent.numel())

        return base_y + delta.reshape(*base_y.shape)

def _get_parent(model,name):
    parts=name.split(".")
    obj=model
    for p in parts[:-1]:
        if p.isdigit():
            obj=obj[int(p)]
        else:
            obj=getattr(obj,p)
    return obj,parts[-1]

def install_scratch_moe(model):
    # Freeze every pretrained Voxtral parameter.
    for p in model.parameters():
        p.requires_grad=False

    candidates=[]
    for name,module in list(model.named_modules()):
        leaf=name.rsplit(".",1)[-1]
        if leaf in TARGETS and hasattr(module,"in_features") and hasattr(module,"out_features"):
            candidates.append((name,module))

    if not candidates:
        raise RuntimeError("No q/k/v/o projection modules found.")

    bank=RouterBank()
    model.moe_router_bank=bank

    # Shape-aware router groups.
    #
    # q/k/v normally share the same input hidden width, so they share one
    # router.  o_proj may consume the concatenated attention-head width,
    # which can differ from the model hidden size under GQA.  In that case
    # it receives a separate router for the SAME attention block.
    #
    # This avoids dimension mismatch while keeping routing as shared as the
    # architecture permits.
    group_to_key={}
    group_members=defaultdict(list)

    for name,module in candidates:
        prefix=name.rsplit(".",1)[0]
        in_dim=int(module.in_features)
        group=(prefix,in_dim)
        group_members[group].append(name)
        if group not in group_to_key:
            key=f"router_{len(group_to_key):03d}"
            group_to_key[group]=key
            dev=next(module.parameters()).device
            router=nn.Linear(
                in_dim,N_EXPERTS,bias=False,
                device=dev,dtype=torch.float32
            )
            nn.init.normal_(router.weight,mean=0.0,std=1e-3)
            bank[key]=router

    # Exactly one projection wrapper per router group records route statistics
    # and contributes the router load-balance / entropy regularizers.  This
    # prevents duplicate regularization for q/k/v that share the same router.
    aux_owner={}
    for group,names in group_members.items():
        # Prefer q_proj when present, otherwise the first module in the group.
        preferred=[n for n in names if n.endswith(".q_proj")]
        aux_owner[group]=preferred[0] if preferred else names[0]

    wrapped=[]
    for name,module in candidates:
        parent,attr=_get_parent(model,name)
        prefix=name.rsplit(".",1)[0]
        group=(prefix,int(module.in_features))
        record_aux=(name==aux_owner[group])
        wrapper=SparseMoELoraLinear(
            module,bank,group_to_key[group],
            n_experts=N_EXPERTS,top_k=TOP_K,rank=RANK,
            alpha=LORA_ALPHA,dropout=LORA_DROPOUT,
            record_aux=record_aux,
        )
        setattr(parent,attr,wrapper)
        wrapped.append(name)

    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    router_params=sum(p.numel() for p in model.moe_router_bank.parameters())
    expert_params=sum(
        p.numel() for n,p in model.named_parameters()
        if p.requires_grad and ("moe_A" in n or "moe_B" in n)
    )
    # Human-readable topology audit.
    topology=[]
    for group,names in sorted(group_members.items(), key=lambda kv: (kv[0][0],kv[0][1])):
        prefix,in_dim=group
        topology.append({
            "attention_prefix":prefix,
            "router_key":group_to_key[group],
            "router_input_dim":in_dim,
            "modules":[n.rsplit(".",1)[-1] for n in names],
            "aux_owner":aux_owner[group].rsplit(".",1)[-1],
        })

    print("=== INTERNAL SCRATCH-MOE INSTALLED ===")
    print("router_group_count:",len(group_to_key))
    print("wrapped_projection_count:",len(wrapped))
    print("n_experts:",N_EXPERTS,"top_k:",TOP_K,"rank:",RANK)
    print("shared_expert:",False)
    print("router_parameters:",router_params)
    print("expert_parameters:",expert_params)
    print("total_trainable_parameters:",trainable)

    dim_counts=defaultdict(int)
    for (prefix,in_dim) in group_to_key:
        dim_counts[in_dim]+=1
    print("router_input_dim_counts:",dict(sorted(dim_counts.items())))

    # Print only the first few groups; full topology is saved in RUN_SUMMARY.
    print("router_topology_preview:")
    for row in topology[:12]:
        print(" ",row)

    return {
        "router_count":len(group_to_key),
        "wrapped_projection_count":len(wrapped),
        "router_parameters":router_params,
        "expert_parameters":expert_params,
        "total_trainable_parameters":trainable,
        "router_input_dim_counts":dict(sorted(dim_counts.items())),
        "router_topology":topology,
        "wrapped_modules":wrapped,
    }

def q_wrappers(model):
    return [m for m in model.modules() if isinstance(m,SparseMoELoraLinear) and m.record_aux]

def reset_route_stats(model):
    for m in q_wrappers(model):
        m.reset_stats()

def moe_aux(model):
    lbs=[]; ents=[]
    for m in q_wrappers(model):
        if m.last_lb_loss is not None: lbs.append(m.last_lb_loss)
        if m.last_entropy is not None: ents.append(m.last_entropy)
    if not lbs:
        z=next(p for p in model.parameters() if p.requires_grad).sum()*0.0
        return z,z
    return torch.stack(lbs).mean(),torch.stack(ents).mean()

def route_summary(model):
    counts=torch.zeros(N_EXPERTS,dtype=torch.long)
    ent_sum=0.0; tokens=0
    for m in q_wrappers(model):
        counts += m.route_counts.detach().cpu()
        ent_sum += float(m.entropy_sum.detach().cpu())
        tokens += int(m.routed_token_count.detach().cpu())
    total=max(1,int(counts.sum()))
    return {
        "expert_selected_counts":[int(x) for x in counts.tolist()],
        "expert_selection_share":[float(x/total) for x in counts.tolist()],
        "normalized_router_entropy":float(ent_sum/max(1,tokens)),
        "average_routed_k":float(counts.sum().item()/max(1,tokens)),
        "all_experts_used":bool((counts>0).all()),
    }

@torch.no_grad()
def infer_one(model,processor,audio_path,device,max_new_tokens=256):
    model.eval()
    x=prompt_inputs(processor,audio_path,device)
    plen=x["input_ids"].shape[1]
    t0=time.time()
    out=model.generate(
        **x,max_new_tokens=max_new_tokens,do_sample=False,
        no_repeat_ngram_size=3,repetition_penalty=1.3
    )
    sec=time.time()-t0
    txt=processor.batch_decode(out[:,plen:],skip_special_tokens=True)[0]
    return clean(txt),sec

@torch.no_grad()
def evaluate(model,processor,df,device,label,out_csv):
    reset_route_stats(model)
    rows=[]; refs=[]; hyps=[]
    print(f"\n=== {label} ===")
    for i,(_,r) in enumerate(df.iterrows(),1):
        g=clean_transcript(r["asr_target_clean"])
        p,sec=infer_one(model,processor,r["resolved_audio_path"],device)
        rw=row_wer(g,p)
        refs.append(g);hyps.append(p)
        rows.append({
            "audio_name":r["audio_name"],
            "reference_transcript":g,
            "predicted_transcript":p,
            "row_wer_ehsan":rw,
            "row_cer":row_cer(g,p),
            "token_f1":token_f1(g,p),
            "runtime_sec":sec,
            "blank":int(not p),
        })
        if i<=5 or i%25==0 or i==len(df):
            print(f"[{i:04d}/{len(df)}] rowWER={rw:.4f} mean={sum(x['row_wer_ehsan'] for x in rows)/len(rows):.4f}")
    d=pd.DataFrame(rows)
    out_csv=Path(out_csv);out_csv.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(out_csv,index=False)
    m={
        "n_rows":len(d),
        "mean_row_WER_Ehsan_style":float(d.row_wer_ehsan.mean()),
        "mean_row_CER":float(d.row_cer.mean()),
        "corpus_WER":float(corpus_wer(refs,hyps)),
        "mean_TokenF1":float(d.token_f1.mean()),
        "mean_runtime_sec":float(d.runtime_sec.mean()),
        "blank_predictions":int(d.blank.sum()),
        **route_summary(model),
    }
    print("METRICS:",json.dumps(m,indent=2))
    return m

def snapshot_trainables(model):
    return {
        n:p.detach().float().cpu().clone()
        for n,p in model.named_parameters()
        if p.requires_grad
    }

def delta_norm(initial,model):
    sq=0.0
    for n,p in model.named_parameters():
        if p.requires_grad and n in initial:
            d=p.detach().float().cpu()-initial[n]
            sq += float((d*d).sum())
    return math.sqrt(sq)

def save_moe_checkpoint(model,outdir,metadata):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    state={
        n:p.detach().cpu()
        for n,p in model.named_parameters()
        if p.requires_grad
    }
    torch.save(state,outdir/"moe_trainable_state.pt")
    (outdir/"moe_config.json").write_text(json.dumps(metadata,indent=2))

def train_moe(
    model,processor,train,dev,device,outdir,
    epochs,expert_lr,router_lr,grad_accum,patience,
    lb_coef,entropy_coef,label,smoke=False
):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    routers=list(model.moe_router_bank.parameters())
    router_ids={id(p) for p in routers}
    experts=[p for p in model.parameters() if p.requires_grad and id(p) not in router_ids]

    opt=AdamW([
        {"params":experts,"lr":expert_lr},
        {"params":routers,"lr":router_lr},
    ],weight_decay=0.01)

    initial=snapshot_trainables(model)
    best=float("inf");best_epoch=0;stale=0;history=[]
    router_grad_steps=0;expert_grad_steps=0
    t0=time.time()

    for ep in range(1,epochs+1):
        model.train();reset_route_stats(model)
        order=train.sample(frac=1.0,random_state=100+ep).reset_index(drop=True)
        opt.zero_grad(set_to_none=True)
        losses=[];task_losses=[];skipped=0
        pb=tqdm(order.iterrows(),total=len(order),desc=f"{label} ep{ep} train",dynamic_ncols=True,mininterval=5)

        for step,(_,r) in enumerate(pb,1):
            b=make_train_batch(
                processor,r["resolved_audio_path"],
                clean_transcript(r["asr_target_clean"]),device
            )
            o=model(**b,use_cache=False,return_dict=True)
            lb,ent=moe_aux(model)
            task=o.loss
            loss=task + lb_coef*lb - entropy_coef*ent

            if not torch.isfinite(loss):
                skipped+=1;opt.zero_grad(set_to_none=True);continue
            (loss/grad_accum).backward()

            if any(p.grad is not None and p.grad.detach().abs().sum().item()>0 for p in routers):
                router_grad_steps+=1
            if any(p.grad is not None and p.grad.detach().abs().sum().item()>0 for p in experts):
                expert_grad_steps+=1

            if step%grad_accum==0 or step==len(order):
                torch.nn.utils.clip_grad_norm_(experts+routers,1.0)
                opt.step();opt.zero_grad(set_to_none=True)

            losses.append(float(loss.detach()))
            task_losses.append(float(task.detach()))
            pb.set_postfix(
                task=f"{task_losses[-1]:.3f}",
                lb=f"{float(lb.detach()):.3f}",
                ent=f"{float(ent.detach()):.3f}"
            )
            del b,o,loss,task,lb,ent

        pb.close();gc.collect();torch.cuda.empty_cache()
        train_route=route_summary(model)
        m=evaluate(model,processor,dev,device,f"{label} DEV EPOCH {ep}",outdir/f"dev_epoch_{ep}.csv")
        rec={
            "epoch":ep,
            "train_total_loss":sum(losses)/max(1,len(losses)),
            "train_task_loss":sum(task_losses)/max(1,len(task_losses)),
            "skipped":skipped,
            "train_route":json.dumps(train_route),
            **m,
        }
        history.append(rec)
        pd.DataFrame(history).to_csv(outdir/"HISTORY.csv",index=False)

        score=m["mean_row_WER_Ehsan_style"]
        if score<best:
            best=score;best_epoch=ep;stale=0
            meta={
                "architecture":"Voxtral-24B internal Scratch-MoE",
                "experts":N_EXPERTS,"top_k":TOP_K,"rank":RANK,
                "alpha":LORA_ALPHA,"dropout":LORA_DROPOUT,
                "shared_expert":False,
                "routing":"fixed_top2",
                "best_epoch":ep,
                "best_metrics":m,
            }
            save_moe_checkpoint(model,outdir/"GLOBAL_BEST",meta)
            print(f"*** NEW GLOBAL BEST {best:.6f} at epoch {ep} ***")
        else:
            stale+=1
            print(f"no improvement {stale}/{patience}; best={best:.6f}")
        if stale>=patience:
            print("EARLY STOP at epoch",ep)
            break

    cert={
        "router_gradient_nonzero_steps":router_grad_steps,
        "expert_gradient_nonzero_steps":expert_grad_steps,
        "full_trainable_delta_l2":delta_norm(initial,model),
        "best_epoch":best_epoch,
        "best_mean_row_WER":best,
        "final_route_summary":route_summary(model),
        "train_runtime_sec":time.time()-t0,
    }
    (outdir/"TRAINING_CERT.json").write_text(json.dumps(cert,indent=2))
    print("TRAINING_CERT:",json.dumps(cert,indent=2))
    return cert

def validate_data(split):
    audit=json.loads((split/"PREPROCESSING_AND_ALIGNMENT_SUMMARY.json").read_text())
    if audit.get("official175_used_for_training_or_selection") is not False:
        raise RuntimeError("official175 exclusion audit failed")
    train=pd.read_csv(split/"train_segment_aligned281.csv",low_memory=False)
    dev=pd.read_csv(split/"dev_segment_aligned61.csv",low_memory=False)
    locked=pd.read_csv(split/"locked_official175_identity_audit_DO_NOT_TRAIN.csv",low_memory=False)
    if set(train.audio_name)&set(locked.audio_name): raise RuntimeError("OFFICIAL175 LEAK IN TRAIN")
    if set(dev.audio_name)&set(locked.audio_name): raise RuntimeError("OFFICIAL175 LEAK IN DEV")
    return train,dev

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split-dir",default=str(Path(__file__).resolve().parent/"data"))
    ap.add_argument("--outdir",default="/data/smgreen1/voxtral_moe_v1/internal_scratch_moe_fixed_top2_v1_2")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--expert-lr",type=float,default=5e-5)
    ap.add_argument("--router-lr",type=float,default=1e-4)
    ap.add_argument("--grad-accum",type=int,default=4)
    ap.add_argument("--patience",type=int,default=2)
    ap.add_argument("--lb-coef",type=float,default=0.01)
    ap.add_argument("--entropy-coef",type=float,default=0.001)
    args=ap.parse_args()

    split=Path(args.split_dir)
    train_df,dev_df=validate_data(split)

    if args.smoke:
        train_df=train_df.head(12).copy()
        dev_df=dev_df.head(6).copy()
        args.epochs=2
        args.grad_accum=2
        args.patience=2
        args.outdir="/data/smgreen1/voxtral_moe_v1/internal_scratch_moe_fixed_top2_SMOKE_v1_2"

    print("=== VOXTRAL INTERNAL SCRATCH-MOE FIXED TOP-2 ===")
    print("mode:","SMOKE" if args.smoke else "FULL")
    print("train:",len(train_df),"dev:",len(dev_df),"official175_used: False")
    print("baseline_to_beat_mean_row_WER: 0.3154370526978759")
    print("experts=4 top_k=2 rank=16 shared_expert=False")
    print("IMPORTANT: experts + routers are newly initialized; Voxtral remains pretrained.")

    model,processor,device=load_base(0)
    install_info=install_scratch_moe(model)

    cert=train_moe(
        model,processor,train_df,dev_df,device,args.outdir,
        args.epochs,args.expert_lr,args.router_lr,args.grad_accum,args.patience,
        args.lb_coef,args.entropy_coef,
        "SCRATCH-MOE-TOP2",smoke=args.smoke
    )

    summary={"install":install_info,"training_cert":cert,"official175_used":False}
    Path(args.outdir).mkdir(parents=True,exist_ok=True)
    (Path(args.outdir)/"RUN_SUMMARY.json").write_text(json.dumps(summary,indent=2))
    print("DONE:",args.outdir)

if __name__=="__main__":
    main()
