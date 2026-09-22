#!/usr/bin/env python3
"""PT104: frozen inference-only 2-group routing prototype, TRAIN56 gate."""
from pathlib import Path
import hashlib,json,traceback
import pandas as pd
import numpy as np
import torch
import pt61_reference as P
from pt102_preflight import check,split_ids,sha,TRAIN,DEV

HERE=Path(__file__).resolve().parent
OUT=Path('/data/smgreen1/voxtral_moe_v1/pt104_frozen_two_group_router_v1_0')
ARCHIVED=HERE/'data/TRAIN56_PT57_ARCHIVED.csv'
BASEDEV=HERE/'data/DEV61_PT57_ACCEPTED_BASELINE.csv'

def paired(a,b,na,nb,out):
    aa=a.set_index('audio_name',verify_integrity=True)
    bb=b.set_index('audio_name',verify_integrity=True)
    if len(aa)!=len(bb) or set(aa.index)!=set(bb.index):
        raise RuntimeError(f'{na}/{nb} identities mismatch')
    if any(str(aa.loc[i,'reference_transcript'])!=str(bb.loc[i,'reference_transcript']) for i in aa.index):
        raise RuntimeError('Reference mismatch between paired runs')
    rows=[]
    for i in aa.index:
        x=float(aa.loc[i,'row_wer_ehsan_style']);y=float(bb.loc[i,'row_wer_ehsan_style'])
        rows.append({'audio_name':i,'baseline_row_wer':x,'candidate_row_wer':y,'delta_candidate_minus_baseline':y-x,
                     'reference_transcript':str(aa.loc[i,'reference_transcript']),
                     'baseline_predicted_transcript':str(aa.loc[i,'predicted_transcript']),
                     'candidate_predicted_transcript':str(bb.loc[i,'predicted_transcript'])})
    p=pd.DataFrame(rows);p.to_csv(out,index=False)
    d=p.delta_candidate_minus_baseline
    return {'baseline_mean_row_WER':float(p.baseline_row_wer.mean()),'candidate_mean_row_WER':float(p.candidate_row_wer.mean()),
            'improved_rows':int((d< -1e-12).sum()),'worsened_rows':int((d>1e-12).sum()),'tied_rows':int((d.abs()<=1e-12).sum())}

def main():
    checks=check()
    OUT.mkdir(parents=True,exist_ok=True)
    t=pd.read_csv(TRAIN,low_memory=False)
    fit,cal=split_ids(t.audio_name.tolist())
    caldf=t.set_index('audio_name',verify_integrity=True).loc[cal].reset_index()
    if len(caldf)!=56 or caldf.resolved_audio_path.map(lambda v:not Path(v).is_file()).any():
        raise RuntimeError('TRAIN56 recording identity/audio gate failed')
    # Compare against PT102's real saved training-cal baseline, independent of source patch.
    saved=pd.read_csv(ARCHIVED)
    if len(saved)!=56 or set(saved.audio_name)!=set(cal):raise RuntimeError('Archived CAL56 identities mismatch')
    print('PT104: loading accepted PT57; no expert/router checkpoint updates',flush=True)
    model,processor,device=P.B.load_base(0)
    P.D.install_dynamic_moe(model)
    pt57sha=P.load_pt57_state(model)
    P.set_accepted_policy()
    P.D.HIERARCHICAL_ENABLED=False
    model.eval()
    baseline,bs=P.evaluate(model,processor,caldf,device,'PT104 TRAIN56 FROZEN PT57 ORIGINAL ROUTER',OUT/'TRAIN56_BASELINE.csv')
    bs.to_csv(OUT/'TRAIN56_BASELINE_CRITICAL_SPANS.csv',index=False)
    baseline_df=pd.read_csv(OUT/'TRAIN56_BASELINE.csv')
    archive_pair=paired(saved,baseline_df,'PT102_ARCHIVE','PT104_PT57',OUT/'TRAIN56_BASELINE_PARITY.csv')
    if any(str(saved.set_index('audio_name').loc[i,'predicted_transcript'])!=str(baseline_df.set_index('audio_name').loc[i,'predicted_transcript']) for i in cal):
        raise RuntimeError('Frozen PT57 TRAIN56 transcript parity failed vs PT102 archive')
    if abs(archive_pair['candidate_mean_row_WER']-0.05771492076888888)>1e-9:
        raise RuntimeError('Frozen PT57 CAL56 WER parity failed vs accepted PT102 run')
    print('PT57 TRAIN56 PARITY PASS',archive_pair['candidate_mean_row_WER'],flush=True)
    P.D.HIERARCHICAL_ENABLED=True
    candidate,cs=P.evaluate(model,processor,caldf,device,'PT104 TRAIN56 TWO-GROUP ROUTER',OUT/'TRAIN56_TWO_GROUP.csv')
    cs.to_csv(OUT/'TRAIN56_TWO_GROUP_CRITICAL_SPANS.csv',index=False)
    newdf=pd.read_csv(OUT/'TRAIN56_TWO_GROUP.csv')
    pp=paired(baseline_df,newdf,'PT57','TWO_GROUP',OUT/'TRAIN56_PAIRED.csv')
    if abs(float(candidate['average_routed_k'])-float(baseline['average_routed_k']))>0.04 or not 1.80<=float(candidate['average_routed_k'])<=1.98:
        raise RuntimeError('Compute-budget parity failed: average routed k changed')
    for key in ('k1_fraction','k2_fraction','k3_fraction'):
        if abs(float(candidate[key])-float(baseline[key]))>0.05:
            raise RuntimeError('Compute-budget parity failed: '+key)
    if sha(P.PT57)!=pt57sha:raise RuntimeError('Accepted PT57 checkpoint changed')
    improved=(float(candidate['mean_row_WER_Ehsan_style'])<float(baseline['mean_row_WER_Ehsan_style'])-1e-12)
    preserved=(candidate['critical_words_correct']>=baseline['critical_words_correct'] and candidate['critical_words_total']==baseline['critical_words_total'])
    pass_gate=bool(improved and preserved and candidate.get('all_experts_used',False))
    frozen={'pt57_sha256':pt57sha,'train_sha256':sha(TRAIN),'dev_sha256':sha(DEV),
            'router_source_sha256':sha(HERE/'source/01_risk_adaptive_dynamic_k.py'),
            'policy':'fixed groups 0,1 and 2,3; primary group by probability mass; same original k thresholds; no model update',
            'selection_set':'TRAIN56 previously included in original PT57 training',
            'train56_gate_pass':pass_gate,'train56_pair':pp,'critical_correct_baseline':baseline['critical_words_correct'],
            'critical_correct_candidate':candidate['critical_words_correct'],'historical175_used':False}
    lockpath=OUT/'POLICY_LOCK_BEFORE_DEV.json';lockpath.write_text(json.dumps(frozen,indent=2))
    print('TRAIN56 ROUTER GATE',json.dumps(frozen,indent=2),flush=True)
    if not pass_gate:
        (OUT/'PT104_RESULT.json').write_text(json.dumps({'status':'TRAIN56_REJECTED_NO_DEV','train56':frozen,'independent_validation':False},indent=2))
        print('PT104: no TRAIN56 gain with preserved critical words; DEV61 not evaluated; PT57 remains accepted.',flush=True)
        return
    # Only now read the development *values*. This cohort has been examined historically.
    devdf=pd.read_csv(DEV,low_memory=False)
    if len(devdf)!=61 or devdf.audio_name.duplicated().any():raise RuntimeError('DEV61 identity check failed')
    if devdf.resolved_audio_path.map(lambda v:not Path(v).is_file()).any():raise RuntimeError('DEV61 audio missing')
    devmet,dsp=P.evaluate(model,processor,devdf,device,'PT104 DEV61 FROZEN TWO-GROUP ONCE',OUT/'DEV61_TWO_GROUP.csv')
    dsp.to_csv(OUT/'DEV61_TWO_GROUP_CRITICAL_SPANS.csv',index=False)
    devpair=paired(pd.read_csv(BASEDEV),pd.read_csv(OUT/'DEV61_TWO_GROUP.csv'),'PT57_ACCEPTED_DEV','PT104_DEV',OUT/'DEV61_PAIRED.csv')
    if sha(P.PT57)!=pt57sha or json.loads(lockpath.read_text())!=frozen:raise RuntimeError('Freeze integrity failed')
    result={'status':'DEV61_EXPLORATORY_COMPLETE','train56':frozen,'dev61_metrics':devmet,'dev61_paired':devpair,
            'historical175_used':False,'independent_validation':False,'accepted_pt57_replacement':False,
            'warning':'fixed arbitrary groups; not a learned hierarchy, cannot establish merits of trained hierarchical MoE'}
    (OUT/'PT104_RESULT.json').write_text(json.dumps(result,indent=2))
    print('PT104_RESULT',json.dumps(result,indent=2),flush=True)
    print('PT57 original remains accepted; no automatic promotion.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc();print('PT104_STOP:',exc,flush=True);raise
