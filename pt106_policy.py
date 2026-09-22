#!/usr/bin/env python3
"""Dependency-light PT105 profile verification + deterministic PT106 policy."""
import numpy as np
import pandas as pd

GROUPS={
 '01_23':((0,1),(2,3)),
 '02_13':((0,2),(1,3)),
 '03_12':((0,3),(1,2)),
}
FRACS={'01_23':'fraction_01_23','02_13':'fraction_02_13','03_12':'fraction_03_12'}
MIN_AFFINITY=0.75
MIN_MARGIN=0.35
MIN_K2PLUS=1000
SECOND_PROB_RATIO=0.95
GATE_N=64
GATE_SEED=10600


def select_group_policy(profile,probe):
    if probe.get('status')!='READ_ONLY_PROFILE_COMPLETE' or not all(probe.get(k) is True for k in ('baseline_transcript_parity','baseline_reference_parity','baseline_wer_parity')):
        raise RuntimeError('PT105 source probe not certified')
    if probe['n_local_router_groups']!=112 or len(profile)!=112 or profile.router_group.duplicated().any():
        raise RuntimeError('expected exactly 112 unique router groups')
    if set(profile.router_group)!={f'router_{i:03d}' for i in range(112)}:
        raise RuntimeError('router names do not match accepted PT57 router-group keys')
    agg={n:0.0 for n in FRACS}
    selected={}
    rows=[]
    for _,r in profile.sort_values('router_group').iterrows():
        counts=[int(r['pair_'+v]) for v in ('01','02','03','12','13','23')]
        total=int(r.k2plus_tokens)
        if total<0 or any(v<0 for v in counts) or sum(counts)!=total:
            raise RuntimeError(f'pair counts invalid for {r.router_group}')
        fr={n:float(r[col]) for n,col in FRACS.items()}
        if any(not np.isfinite(x) or x<0 or x>1 for x in fr.values()) or abs(sum(fr.values())-1)>1e-6:
            raise RuntimeError('invalid group pairing fractions')
        for n in FRACS:
            agg[n]+=total*fr[n]
        order=sorted(fr,key=lambda n:(-fr[n],n))
        choice=order[0];margin=fr[choice]-fr[order[1]]
        enabled=bool(fr[choice]>=MIN_AFFINITY and margin>=MIN_MARGIN and total>=MIN_K2PLUS)
        if enabled:
            mapping=[None]*4
            for a,b in GROUPS[choice]:mapping[a]=b;mapping[b]=a
            assert sorted(mapping)==[0,1,2,3] and all(mapping[mapping[i]]==i and mapping[i]!=i for i in range(4))
            selected[str(r.router_group)]=tuple(mapping)
        rows.append({'router_group':str(r.router_group),'selected':enabled,'pairing':choice if enabled else 'original_unrestricted',
                     'affinity':fr[choice],'margin':margin,'k2plus_tokens':total})
    src=probe['top2_coactivation']
    if int(src['total_k2plus']) != int(profile.k2plus_tokens.sum()):
        raise RuntimeError('PT105 top2 global total mismatch')
    for n in FRACS:
        if abs(agg[n]/float(src['total_k2plus'])-float(src['pairing_coverage'][n]['same_pair_fraction']))>1e-9:
            raise RuntimeError('PT105 global pairing fraction mismatch: '+n)
    return selected,pd.DataFrame(rows)


