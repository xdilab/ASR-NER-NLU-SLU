#!/usr/bin/env python3
"""PT108: matched learned Omni-style residual vs two-level logit router.
Not a full Omni-Router or conditional hierarchical MoE; original PT57 top-k remains.
TRAIN225 fit, TRAIN56 selection, exploratory DEV61 ONLY if a candidate clears
TRAIN56 baseline WER + critical-word + matched compute gates. Never official175.
"""
from __future__ import annotations
import gc, hashlib, json, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm
import pt61_reference as P
from pt102_preflight import check, split_ids, sha, TRAIN, DEV
from pt108_adapters import make_bank

ROOT=Path(__file__).resolve().parent
OUT=Path('/data/smgreen1/voxtral_moe_v1/pt108_matched_learned_router_v1_0')
SOURCE=ROOT/'source/01_risk_adaptive_dynamic_k.py'
SOURCE_UNPATCHED_SHA='9a726c82fc753c11ab49407b8a11ccd2e3a7af777b3ee164d546b8762a18e881'
BASE56=ROOT/'data/PT105_TRAIN56_BASELINE.csv'
BASEDEV=ROOT/'data/DEV61_PT57_ACCEPTED_BASELINE.csv'
MODES=('omni','hierarchical')
EPOCHS=1
LEARNING_RATE=2e-5
GRAD_ACCUM=4
LB_COEF=0.001
KL_COEF=0.02

def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n')

def original_params(model):
    # Only ~10M PT57 expert/router/head params; NEVER clone the 24B pretrained base.
    return [(n,p) for n,p in model.named_parameters()
            if n.endswith('.moe_A') or n.endswith('.moe_B')
            or n.startswith('moe_router_bank.') or n.startswith('moe_criticality_bank.')]

def snapshot_original(model):
    return {n:p.detach().cpu().clone() for n,p in original_params(model)}

def verify_original(model,snapshot):
    pairs=original_params(model)
    if set(n for n,_ in pairs)!=set(snapshot):raise RuntimeError('Original parameter set changed')
    mismatched=[n for n,p in pairs if not torch.equal(p.detach().cpu(),snapshot[n])]
    if mismatched:raise RuntimeError('Original PT57 weights changed: '+repr(mismatched[:5]))

def disable(model):
    P.D.PT108_ADAPTER_BANK=None
    if hasattr(model,'pt108_adapter_bank'):
        delattr(model,'pt108_adapter_bank')
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def install(model,mode):
    disable(model)
    bank,pairing,count=make_bank(model,mode)
    model.add_module('pt108_adapter_bank',bank)
    P.D.PT108_ADAPTER_BANK=bank
    params=list(bank.parameters())
    for n,p in model.named_parameters():p.requires_grad_(n.startswith('pt108_adapter_bank.'))
    if sum(p.numel() for p in params)!=count or any(p.requires_grad for n,p in model.named_parameters() if not n.startswith('pt108_adapter_bank.')):
        raise RuntimeError('Parameter isolation failed')
    return bank,pairing,params,count

def assert_parity(recording,reference):
    current=pd.read_csv(recording).set_index('audio_name',verify_integrity=True)
    archive=pd.read_csv(reference).set_index('audio_name',verify_integrity=True)
    if not set(current.index).issubset(archive.index):raise RuntimeError('Baseline identities do not match archived PT57')
    for name in current.index:
        c=current.loc[name];r=archive.loc[name]
        if str(c.predicted_transcript)!=str(r.predicted_transcript) or str(c.reference_transcript)!=str(r.reference_transcript):
            raise RuntimeError('PT57 baseline transcript parity fails: '+name)
        if abs(float(c.row_wer_ehsan_style)-float(r.row_wer_ehsan_style))>1e-10:
            raise RuntimeError('PT57 baseline WER parity fails: '+name)
    return len(current)

def paired(reference,candidate,outpath):
    b=pd.read_csv(reference).set_index('audio_name',verify_integrity=True)
    c=pd.read_csv(candidate).set_index('audio_name',verify_integrity=True)
    if set(b.index)!=set(c.index):raise RuntimeError('Paired IDs differ')
    rows=[]
    for key,r in c.iterrows():
        q=b.loc[key]
        if str(q.reference_transcript)!=str(r.reference_transcript):raise RuntimeError('Reference mismatch: '+key)
        w0=float(q.row_wer_ehsan_style);w1=float(r.row_wer_ehsan_style)
        rows.append({'audio_name':key,'baseline_predicted_transcript':q.predicted_transcript,
            'candidate_predicted_transcript':r.predicted_transcript,'baseline_wer':w0,
            'candidate_wer':w1,'delta_candidate_minus_baseline':w1-w0})
    d=pd.DataFrame(rows);d.to_csv(outpath,index=False)
    delta=d.delta_candidate_minus_baseline
    return {'n':len(d),'baseline_mean_row_wer':float(d.baseline_wer.mean()),
            'candidate_mean_row_wer':float(d.candidate_wer.mean()),
            'improved':int((delta< -1e-12).sum()),'worsened':int((delta>1e-12).sum()),
            'tied':int((delta.abs()<=1e-12).sum())}

def load_adapter(bank,path):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    bank.load_state_dict(obj,strict=True)

def save_adapter(bank,path):
    obj={k:v.detach().cpu().clone() for k,v in bank.state_dict().items()}
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(obj,path)
    return sha(path)

def train_one(model,processor,device,fit,cal,base,baseline_critical,mode,original_snapshot,pt57sha):
    modeout=OUT/mode;modeout.mkdir(parents=True,exist_ok=True)
    P.set_accepted_policy()
    bank,pairing,params,count=install(model,mode)
    write_json(modeout/'ARCHITECTURE_LOCK_BEFORE_TRAIN.json',
       {'mode':mode,'parameter_count':count,'pairing':pairing,'pt57_sha256':pt57sha,
        'train_file_sha256':sha(TRAIN),'dev_file_sha256_only':sha(DEV),
        'adapter_init':'all-zero; full PT57 baseline output exactly at initialization',
        'learning_rate':LEARNING_RATE,'fit_rows':225,'calibration_rows':56,'epochs':EPOCHS,
        'grad_accum':GRAD_ACCUM,'lb_coef':LB_COEF,'kl_coef':KL_COEF,
        'source_sha256':sha(SOURCE),'historical175_used':False,
        'architecture_scope':'learned residual router-logit prototype; unchanged PT57 top-k and experts'})
    # Verify exact PT57 two-row decode parity for each zero-initialized head.
    model.eval()
    tiny=cal.iloc[:2].reset_index(drop=True)
    P.evaluate(model,processor,tiny,device,f'PT108 {mode} ZERO-ADAPTER PARITY',modeout/'ZERO_ADAPTER_PARITY.csv')
    assert_parity(modeout/'ZERO_ADAPTER_PARITY.csv',BASE56)
    print('PT108',mode,'ZERO_ADAPTER_PARITY_PASS',flush=True)
    optimizer=AdamW(params,lr=LEARNING_RATE,weight_decay=0.0)
    history=[];gradient_found=False;skipped=0
    for epoch in range(1,EPOCHS+1):
        model.train();P.D.DYNAMIC_ENABLED=True;P.D.CURRENT_CRIT_TARGET=None
        order=fit.sample(frac=1,random_state=10800+epoch).reset_index(drop=True)
        optimizer.zero_grad(set_to_none=True);accum=0;losses=[]
        progress=tqdm(order.iterrows(),total=len(order),desc=f'PT108 {mode} FIT225 e{epoch}',mininterval=5,dynamic_ncols=True)
        for _,r in progress:
            owners=P.D.wrappers(model)
            for w in owners:
                w.last_lb_loss=None;w.last_pt108_kl_loss=None
            batch=P.B.make_train_batch(processor,r.resolved_audio_path,P.B.clean_transcript(r.asr_target_clean),device)
            output=model(**batch,use_cache=False,return_dict=True)
            lbs=[w.last_lb_loss for w in owners if w.last_lb_loss is not None]
            kls=[w.last_pt108_kl_loss for w in owners if getattr(w,'last_pt108_kl_loss',None) is not None]
            if not lbs or len(lbs)!=len(kls):
                raise RuntimeError(f'No matching local-group auxiliary losses: LB={len(lbs)} KL={len(kls)}')
            lb=torch.stack(lbs).mean();kl=torch.stack(kls).mean()
            loss=output.loss+LB_COEF*lb+KL_COEF*kl
            if not bool(torch.isfinite(loss)):
                skipped+=1
                raise RuntimeError('Nonfinite router loss; stop instead of silent skipping')
            (loss/GRAD_ACCUM).backward();accum+=1
            if not gradient_found:
                nonzero=sum(int(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).any())) for p in params)
                if nonzero==0:raise RuntimeError('ASR/router objective has no nonzero gradient; stop')
                gradient_found=True
                print(f'PT108 {mode} nonzero trainable parameter gradient tensors: {nonzero}',flush=True)
            losses.append(float(loss.detach().cpu()))
            del batch,output,loss,lb,kl,lbs,kls
            for w in owners:
                w.last_lb_loss=None;w.last_pt108_kl_loss=None
            if accum==GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(params,1.0)
                optimizer.step();optimizer.zero_grad(set_to_none=True);accum=0
        if accum:
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        progress.close()
        model.eval();gc.collect();torch.cuda.empty_cache()
        verify_original(model,original_snapshot)
        # FIT225 (not CAL56/DEV61) sets compute thresholds for each trained router.
        budget=P.T.exact_budget_calibrate(model,processor,fit,device,
            cal_rows=32,cert_rows=16,iters=1,seed_base=10810)
        write_json(modeout/'TRAIN225_BUDGET.json',budget)
        if not bool(budget.get('cert_pass')):
            print('PT108',mode,'budget certificate failed; evaluating CAL only for diagnostic, not selecting.',flush=True)
        metrics,spans=P.evaluate(model,processor,cal,device,f'PT108 {mode} TRAIN56',modeout/'TRAIN56_CANDIDATE.csv')
        spans.to_csv(modeout/'TRAIN56_CANDIDATE_CRITICAL_SPANS.csv',index=False)
        assert set(pd.read_csv(modeout/'TRAIN56_CANDIDATE.csv').audio_name)==set(cal.audio_name)
        pair=paired(BASE56,modeout/'TRAIN56_CANDIDATE.csv',modeout/'TRAIN56_PAIRED.csv')
        average_k=float(metrics['average_routed_k'])
        baseline_k=float(base['average_routed_k'])
        critical_ok=(int(metrics['critical_words_total'])==int(base['critical_words_total'])
                      and int(metrics['critical_words_correct'])>=int(baseline_critical))
        better=(float(metrics['mean_row_WER_Ehsan_style'])<float(base['mean_row_WER_Ehsan_style'])-1e-12)
        valid=bool(budget.get('cert_pass')) and abs(average_k-baseline_k)<=0.04 and critical_ok and better and gradient_found
        adapter_path=modeout/'TRAINED_ADAPTER.pt'
        adapter_sha=save_adapter(bank,adapter_path)
        rec={'mode':mode,'epoch':epoch,'status':'PASS' if valid else 'REJECT',
             'train225_avg_loss':float(np.mean(losses)),'train225_skipped':skipped,
             'nonzero_gradients_verified':gradient_found,'critical_preserved':critical_ok,
             'better_train56_wer':better,'compute_budget_within_0_04':abs(average_k-baseline_k)<=0.04,
             'baseline_train56_wer':float(base['mean_row_WER_Ehsan_style']),
             'candidate_train56_wer':float(metrics['mean_row_WER_Ehsan_style']),
             'baseline_critical_words_correct':int(baseline_critical),
             'candidate_critical_words_correct':int(metrics['critical_words_correct']),
             'cal56_paired':pair,'train225_budget_cert':bool(budget.get('cert_pass')),
             'tau1':float(P.D.TAU1),'tau2':float(P.D.TAU2),
             'adapter_sha256':adapter_sha,'adapter_path':str(adapter_path),
             'adapter_trainable_params':count,'mean_active_experts':average_k}
        history.append(rec)
        write_json(modeout/'TRAIN56_GATE.json',rec)
        print('PT108',mode,'TRAIN56_GATE',json.dumps({k:v for k,v in rec.items() if k not in ('cal56_paired',)},indent=2),flush=True)
    verify_original(model,original_snapshot)
    disable(model)
    return history[-1]

def main():
    pre=check(); OUT.mkdir(parents=True,exist_ok=True)
    # Synthetic router test and exact source patch are verified before GPU allocation.
    original_source_bytes=(ROOT/'source/01_risk_adaptive_dynamic_k.py').read_text()
    if original_source_bytes.count('PT108_ADAPTER_BANK[self.router_key](xf)')!=1:
        raise RuntimeError('PT108 source adapter patch absent/duplicated')
    write_json(OUT/'PT108_RUN_CONFIG.json',{'experiment':'PT108','modes':MODES,'train_fit':225,'selection':56,
        'dev61_selection':False,'historical175_used':False,'epochs':EPOCHS,
        'lr':LEARNING_RATE,'grad_accum':GRAD_ACCUM,'lb':LB_COEF,'kl':KL_COEF,
        'pt57_original_source_sha256':SOURCE_UNPATCHED_SHA,'pt108_patched_source_sha256':sha(SOURCE),
        'split_sha256':pre['split_identity_sha256'],
        'warning':'PT57 originally trained on all TRAIN281 and DEV61 has been repeatedly studied; any positive DEV finding is exploratory.'})
    d=pd.read_csv(TRAIN,low_memory=False)
    fitids,calids=split_ids(d.audio_name.tolist())
    indexed=d.set_index('audio_name',verify_integrity=True)
    fit=indexed.loc[fitids].reset_index();cal=indexed.loc[calids].reset_index()
    if len(fit)!=225 or len(cal)!=56 or set(fitids)&set(calids):raise RuntimeError('Split leak')
    model,processor,device=P.B.load_base(0)
    P.D.install_dynamic_moe(model)
    pt57sha=P.load_pt57_state(model)
    P.set_accepted_policy();P.D.PT108_ADAPTER_BANK=None
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    originals=snapshot_original(model)
    # Full TRAIN56 baseline once, decode and score matched to archived PT105 for validity.
    base,spans=P.evaluate(model,processor,cal,device,'PT108 PT57 TRAIN56 BASELINE',OUT/'TRAIN56_PT57_BASELINE.csv')
    spans.to_csv(OUT/'TRAIN56_PT57_BASELINE_CRITICAL_SPANS.csv',index=False)
    assert_parity(OUT/'TRAIN56_PT57_BASELINE.csv',BASE56)
    print('PT108 FULL TRAIN56 PT57 BASELINE PARITY PASS',flush=True)
    if len(cal)!=56 or abs(float(base['mean_row_WER_Ehsan_style'])-0.0577149208)>1e-7:
        raise RuntimeError('TRAIN56 PT57 baseline score does not match accepted PT105')
    gate=[]
    for mode in MODES:
        gate.append(train_one(model,processor,device,fit,cal,base,int(base['critical_words_correct']),mode,originals,pt57sha))
    verify_original(model,originals)
    valid=[r for r in gate if r['status']=='PASS']
    result={'status':'TRAIN56_NO_ACCEPTED_ROUTER','pt57_sha256':pt57sha,'train56_baseline':base,
       'train56_modes':gate,'dev61_candidate_evaluated':False,'historical175_used':False,
       'pt57_replaced':False,'independent_validation':False}
    if not valid:
        write_json(OUT/'PT108_RESULT.json',result)
        print('PT108 TRAIN56 GATE: neither learned router passed; DEV61 NOT READ. Keep PT57.',flush=True)
        return
    # Pick winner via TRAIN56 ONLY; freeze weights, thresholds, identity, and source hash.
    valid.sort(key=lambda r:(r['candidate_train56_wer'],-r['candidate_critical_words_correct'],r['mode']))
    chosen=valid[0]
    mode=chosen['mode']
    bank,pairing,count=install(model,mode)
    load_adapter(bank,Path(chosen['adapter_path']))
    P.D.TAU1=chosen['tau1'];P.D.TAU2=chosen['tau2'];P.D.DYNAMIC_ENABLED=True
    frozen={'mode':mode,'adapter_sha256':chosen['adapter_sha256'],'pt57_sha256':pt57sha,
      'source_sha256':sha(SOURCE),'train_source_sha256':sha(TRAIN),'dev_source_sha256':sha(DEV),
      'train56_winner_metric':chosen['candidate_train56_wer'],'tau1':chosen['tau1'],'tau2':chosen['tau2'],
      'hierarchical_pairing':pairing if mode=='hierarchical' else None,
      'selection':'TRAIN56 ONLY, before opening DEV61 gold','historical175_used':False}
    write_json(OUT/'FROZEN_BEFORE_DEV61.json',frozen)
    if sha(Path(chosen['adapter_path']))!=chosen['adapter_sha256']:
        raise RuntimeError('Selected adapter hash changed')
    verify_original(model,originals)
    print('PT108 TRAIN56-selected frozen winner',mode,'adapter sha',chosen['adapter_sha256'],flush=True)
    # This is an exploratory historical development evaluation, not pristine independent test.
    dev=pd.read_csv(DEV,low_memory=False)
    if len(dev)!=61 or dev.audio_name.duplicated().any() or dev.resolved_audio_path.map(lambda v:not Path(v).is_file()).any():
        raise RuntimeError('DEV61 identity/audio gate failed')
    model.eval()
    metrics,sp=P.evaluate(model,processor,dev,device,'PT108 TRAIN56-selected DEV61 FROZEN ONCE',OUT/'DEV61_WINNER.csv')
    sp.to_csv(OUT/'DEV61_WINNER_CRITICAL_SPANS.csv',index=False)
    pair=paired(BASEDEV,OUT/'DEV61_WINNER.csv',OUT/'DEV61_PAIRED.csv')
    verify_original(model,originals)
    if sha(Path(chosen['adapter_path']))!=frozen['adapter_sha256'] or sha(SOURCE)!=frozen['source_sha256']:
        raise RuntimeError('Frozen checkpoint or code changed during DEV evaluation')
    result.update({'status':'DEV61_EXPLORATORY_COMPLETE','winner':mode,'winner_cal56':chosen,
       'dev61_winner_metrics':metrics,'dev61_paired':pair,'dev61_candidate_evaluated':True,
       'independent_validation':False,'pt57_replaced':False,
       'note':'Learned 4D residual router-logit prototypes; not a full Omni Router or full two-level dispatch hierarchy.'})
    write_json(OUT/'PT108_RESULT.json',result)
    print('PT108 DEV61 paired result',json.dumps(pair,indent=2),flush=True)
    print('PT57 remains accepted until independent testing.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc();print('PT108_STOP',repr(exc),flush=True);raise
