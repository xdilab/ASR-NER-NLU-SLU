#!/usr/bin/env python3
"""PT106: conservative local expert-pair routing, PT105 profile only, no training."""
from pathlib import Path
import hashlib,json,traceback
import pandas as pd
import numpy as np
import pt61_reference as P
from pt102_preflight import check,split_ids,sha,TRAIN,DEV

HERE=Path(__file__).resolve().parent
OUT=Path('/data/smgreen1/voxtral_moe_v1/pt106_selective_affinity_routing_v1_0')
PROFILE=HERE/'data/PT105_TRAIN56_PER_ROUTER_GROUP_PAIR_PROFILE.csv'
PROBE=HERE/'data/PT105_RESULT.json'
PT105_BASE=HERE/'data/PT105_TRAIN56_BASELINE.csv'
BASEDEV=HERE/'data/DEV61_PT57_ACCEPTED_BASELINE.csv'
from pt106_policy import (select_group_policy,GROUPS,FRACS,MIN_AFFINITY,MIN_MARGIN,
                         MIN_K2PLUS,SECOND_PROB_RATIO,GATE_N,GATE_SEED)

def paired(a,b,out):
    aa=a.set_index('audio_name',verify_integrity=True)
    bb=b.set_index('audio_name',verify_integrity=True)
    if len(aa)!=len(bb) or set(aa.index)!=set(bb.index):raise RuntimeError('paired cohort identities differ')
    rows=[]
    for key in aa.index:
        x=aa.loc[key];y=bb.loc[key]
        if str(x.reference_transcript)!=str(y.reference_transcript):raise RuntimeError('paired gold reference mismatch')
        w1=float(x.row_wer_ehsan_style);w2=float(y.row_wer_ehsan_style)
        rows.append({'audio_name':key,'reference_transcript':str(x.reference_transcript),
           'baseline_predicted_transcript':str(x.predicted_transcript),'candidate_predicted_transcript':str(y.predicted_transcript),
           'baseline_wer':w1,'candidate_wer':w2,'delta_wer_candidate_minus_baseline':w2-w1})
    df=pd.DataFrame(rows);df.to_csv(out,index=False)
    d=df.delta_wer_candidate_minus_baseline
    return {'n':len(df),'baseline_mean_row_wer':float(df.baseline_wer.mean()),
        'candidate_mean_row_wer':float(df.candidate_wer.mean()),'improved':int((d< -1e-12).sum()),
        'worsened':int((d>1e-12).sum()),'tied':int((d.abs()<=1e-12).sum()),
        'largest_single_row_regression':float(d.max())}


def enforce_budget(base,candidate):
    if abs(float(base['average_routed_k'])-float(candidate['average_routed_k']))>0.04:
        raise RuntimeError('average-k compute budget drift > 0.04')
    for field in ('k1_fraction','k2_fraction','k3_fraction'):
        if abs(float(base[field])-float(candidate[field]))>0.05:
            raise RuntimeError(field+' budget drift > 0.05')


def main():
    check()  # DEV CSV is hashed, not read.
    OUT.mkdir(parents=True,exist_ok=True)
    probe=json.loads(PROBE.read_text())
    if probe['pt57_sha256']!=P.EXPECTED_SHA or probe['train56_identity_sha256']!=sha(TRAIN) or probe['dev61_file_sha256_checked_without_reading_labels']!=sha(DEV):
        raise RuntimeError('PT105 input or PT57 hash mismatch')
    profile=pd.read_csv(PROFILE)
    selected,policy_table=select_group_policy(profile,probe)
    if not selected:raise RuntimeError('PT105 yielded zero qualified high-affinity groups')
    policy_table.to_csv(OUT/'TRAIN56_GROUP_SELECTION.csv',index=False)
    original=pd.read_csv(PT105_BASE)
    if len(original)!=56 or original.audio_name.duplicated().any():raise RuntimeError('PT105 saved prediction cohort invalid')
    train=pd.read_csv(TRAIN,low_memory=False)
    fit,cal=split_ids(train.audio_name.tolist())
    if set(cal)!=set(original.audio_name):raise RuntimeError('TRAIN56 original parity cohort differs')
    gate_ids=pd.Series(fit).sample(n=GATE_N,random_state=GATE_SEED).tolist()
    if len(set(gate_ids))!=GATE_N or set(gate_ids)&set(cal):raise RuntimeError('policy-fit and gate cohort overlap')
    train_index=train.set_index('audio_name',verify_integrity=True)
    gate=train_index.loc[gate_ids].reset_index()
    smoke=train_index.loc[original.audio_name.tolist()[:2]].reset_index()
    if any(not Path(v).is_file() for v in gate.resolved_audio_path.tolist()+smoke.resolved_audio_path.tolist()):
        raise RuntimeError('missing gate/smoke audio')
    lock={'pt57_sha256':P.EXPECTED_SHA,'train_sha256':sha(TRAIN),'dev_sha256':sha(DEV),
      'pt105_profile_sha256':sha(PROFILE),'pt105_result_sha256':sha(PROBE),
      'local_router_source_sha256':sha(HERE/'source/01_risk_adaptive_dynamic_k.py'),
      'policy_fit':'PT105 TRAIN56 top-2 coactivation ONLY; no reference values used to select groups',
      'group_affinity_threshold':MIN_AFFINITY,'margin_threshold':MIN_MARGIN,'min_k2plus':MIN_K2PLUS,
      'second_probability_ratio':SECOND_PROB_RATIO,'eligible_router_groups':len(selected),
      'partner_index_per_group':{key:list(val) for key,val in sorted(selected.items())},
      'gate64_ids_sha256':hashlib.sha256(('\n'.join(gate_ids)).encode()).hexdigest(),
      'gate64_from':'TRAIN225 (disjoint from PT105 TRAIN56, but originally used to train PT57)',
      'dev61_untouched_until_gate':True,'historical175_used':False}
    lockpath=OUT/'POLICY_LOCK_BEFORE_SCORING.json'
    lockpath.write_text(json.dumps(lock,indent=2,sort_keys=True))
    print('PT106 FROZEN PREDECLARED POLICY:',json.dumps({k:v for k,v in lock.items() if k!='partner_index_per_group'},indent=2),flush=True)
    print('PT106 selected router groups',len(selected),'of 112',flush=True)
    model,processor,device=P.B.load_base(0)
    P.D.install_dynamic_moe(model)
    ck=P.load_pt57_state(model)
    P.set_accepted_policy()
    P.D.HIERARCHICAL_ENABLED=False
    P.D.PT106_ENABLED=False
    P.D.PT106_GROUP_PAIRS={}
    P.D.PT106_SECOND_PROB_RATIO=SECOND_PROB_RATIO
    model.eval()
    # Audit original PT57 parity using a couple of PT105 reference predictions.
    sm,sp=P.evaluate(model,processor,smoke,device,'PT106 ORIGINAL PT57 SMOKE PARITY',OUT/'TRAIN56_ORIGINAL_SMOKE.csv')
    sm_df=pd.read_csv(OUT/'TRAIN56_ORIGINAL_SMOKE.csv').set_index('audio_name')
    orig=original.set_index('audio_name')
    for key in sm_df.index:
        if str(sm_df.loc[key,'predicted_transcript'])!=str(orig.loc[key,'predicted_transcript']) or abs(float(sm_df.loc[key,'row_wer_ehsan_style'])-float(orig.loc[key,'row_wer_ehsan_style']))>1e-10:
            raise RuntimeError('PT57 unchanged-decoder smoke parity failed')
    print('PT57 SMOKE PARITY PASS',flush=True)
    base,sp=P.evaluate(model,processor,gate,device,'PT106 TRAIN64 ORIGINAL PT57',OUT/'TRAIN64_BASELINE.csv')
    sp.to_csv(OUT/'TRAIN64_BASELINE_CRITICAL_SPANS.csv',index=False)
    P.D.PT106_GROUP_PAIRS=selected
    P.D.PT106_ENABLED=True
    cand,sp=P.evaluate(model,processor,gate,device,'PT106 TRAIN64 LOCAL PAIR ROUTING',OUT/'TRAIN64_CANDIDATE.csv')
    sp.to_csv(OUT/'TRAIN64_CANDIDATE_CRITICAL_SPANS.csv',index=False)
    pair=paired(pd.read_csv(OUT/'TRAIN64_BASELINE.csv'),pd.read_csv(OUT/'TRAIN64_CANDIDATE.csv'),OUT/'TRAIN64_PAIRED.csv')
    sw=sum(int(w.pt106_swap_count.item()) for w in P.D.wrappers(model))
    eligible=sum(int(w.pt106_eligible_count.item()) for w in P.D.wrappers(model))
    enforce_budget(base,cand)
    if sha(P.PT57)!=ck or sha(HERE/'source/01_risk_adaptive_dynamic_k.py')!=lock['local_router_source_sha256'] or json.loads(lockpath.read_text())!=lock:
        raise RuntimeError('policy/checkpoint changed during TRAIN64 gate')
    better=cand['mean_row_WER_Ehsan_style']<base['mean_row_WER_Ehsan_style']-1e-12
    crit=(cand['critical_words_total']==base['critical_words_total'] and cand['critical_words_correct']>=base['critical_words_correct'])
    pass_gate=bool(better and crit and sw>0 and cand.get('all_experts_used',False))
    gate_result={'status':'PASS' if pass_gate else 'FAIL','train64_paired':pair,
       'baseline':base,'candidate':cand,'critical_preserved':crit,'swapped_local_routes':sw,
       'eligible_group_routes':eligible,'selected_router_groups':len(selected),
       'criteria':'strict mean-row WER improvement + no critical-word loss + nonzero swaps + matched budget + all experts used',
       'gate64_in_PT57_original_training':True}
    (OUT/'TRAIN64_GATE.json').write_text(json.dumps(gate_result,indent=2))
    print('PT106 TRAIN64 GATE:',json.dumps({k:v for k,v in gate_result.items() if k not in ('baseline','candidate')},indent=2),flush=True)
    if not pass_gate:
        result={'status':'TRAIN64_REJECTED_NO_DEV','train64':gate_result,'policy_lock':lock,
                'pt57_replaced':False,'historical175_used':False,'independent_validation':False}
        (OUT/'PT106_RESULT.json').write_text(json.dumps(result,indent=2))
        print('PT106: TRAIN64 gate FAILED. DEV61 NOT READ. Keep PT57.',flush=True)
        return
    # Freeze the winning policy BEFORE opening DEV labels; no change to group selection based on gate results.
    (OUT/'POLICY_SELECTED_BEFORE_DEV.json').write_text(json.dumps({'lock':lock,'train64_pass':True,'gate64_paired':pair},indent=2))
    dev=pd.read_csv(DEV,low_memory=False)
    if len(dev)!=61 or dev.audio_name.duplicated().any() or dev.resolved_audio_path.map(lambda v:not Path(v).is_file()).any():
        raise RuntimeError('DEV61 identities/audio check failed')
    dm,ds=P.evaluate(model,processor,dev,device,'PT106 DEV61 FROZEN ONCE',OUT/'DEV61_CANDIDATE.csv')
    ds.to_csv(OUT/'DEV61_CANDIDATE_CRITICAL_SPANS.csv',index=False)
    dp=paired(pd.read_csv(BASEDEV),pd.read_csv(OUT/'DEV61_CANDIDATE.csv'),OUT/'DEV61_PAIRED.csv')
    if sha(P.PT57)!=ck or json.loads(lockpath.read_text())!=lock:raise RuntimeError('freeze integrity failed')
    result={'status':'DEV61_EXPLORATORY_COMPLETE','train64':gate_result,'dev61_paired':dp,'dev61_metrics':dm,
           'policy_lock':lock,'pt57_replaced':False,'historical175_used':False,'independent_validation':False,
           'warning':'DEV61 has been explored in earlier experiments; this is an inference-only local routing prototype, not a trained hierarchical MoE.'}
    (OUT/'PT106_RESULT.json').write_text(json.dumps(result,indent=2))
    print('PT106 DEV61 result:',json.dumps(dp,indent=2),flush=True)
    print('PT57 remains accepted until further review.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        traceback.print_exc();print('PT106_STOP',repr(e),flush=True);raise
