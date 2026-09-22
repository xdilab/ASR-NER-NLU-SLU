#!/usr/bin/env python3
"""CPU-only PT102 preflight. Never opens development reference values."""
from pathlib import Path
import csv, hashlib, json, os, shutil, sys
ROOT=Path(__file__).resolve().parent
TRAIN=ROOT/'data/train281_asr_target_clean_pt57_v2.csv'
DEV=ROOT/'data/dev61_asr_target_clean_pt57_v2.csv'
PT57=Path('/data/smgreen1/voxtral_moe_v1/pt57_final_asr_winner_v1_0/GLOBAL_BEST/moe_trainable_state.pt')
BUDGET=Path('/data/smgreen1/voxtral_moe_v1/pt57g_asr_specialized_dynamic_k_recovery_v1_0/PRIMARY_LR5E5/INITIAL_BUDGET_CERT.json')
EXPECTED={'train': 'c9fa09cd59be3c55be4cc936ee1892a1cfaa80dc124d404c65001c1c98ff3bd7', 'dev': '914399323307bb7cb63b55a423b66f1697aebe61e23a89b25df1c93030a90362', 'pt57':'d44f3708e922f1a6631327c0b24f2e2321b68b1efdc6724752effa477a4e35b6'}
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for ch in iter(lambda:f.read(1<<20), b''):h.update(ch)
 return h.hexdigest()
def split_ids(names):
 # Exactly the PT83 split. Stable when pandas is available; training repeats this split.
 import pandas as pd
 v=pd.Series(names).sample(frac=1.0,random_state=2600).tolist()
 return v[:225],v[225:]
def check():
 for name,p in [('train',TRAIN),('dev',DEV),('pt57',PT57)]:
  if not p.is_file():raise RuntimeError(f'MISSING {name} {p}')
  got=sha(p)
  if got != EXPECTED[name]:raise RuntimeError(f'{name} SHA MISMATCH {got} != {EXPECTED[name]}')
  print(name.upper(),'SHA PASS',got,flush=True)
 if not BUDGET.is_file() or not json.loads(BUDGET.read_text()).get('cert_pass'):
  raise RuntimeError('Accepted PT57 budget certificate missing or not PASS')
 with TRAIN.open(newline='') as f:
  r=csv.DictReader(f)
  names=[]; paths=[]
  for row in r:
   names.append(row['audio_name']);paths.append(row['resolved_audio_path'])
 if len(names)!=281 or len(set(names))!=281:raise RuntimeError('TRAIN281 identity gate FAIL')
 # DEV references remain unopened during preflight: frozen file digest is sufficient.
 missing=[p for p in paths if not Path(p).is_file()]
 if missing:raise RuntimeError(f'{len(missing)} TRAIN audio clips missing, first: {missing[0]}')
 fit,cal=split_ids(names)
 if len(fit)!=225 or len(cal)!=56 or set(fit)&set(cal):raise RuntimeError('FIT/CAL split invalid')
 known=set((ROOT/'data/PT83_CAL56_IDENTITIES.txt').read_text().splitlines())
 if set(cal)!=known:raise RuntimeError(f'CAL56 identity mismatch vs earlier PT83 split: overlap={len(set(cal)&known)}')
 summary={'experiment':'PT102','fit_rows':len(fit),'cal_rows':len(cal),
  'split_seed':2600,'split_identity_sha256':hashlib.sha256(('\\n'.join(fit)+'\\n--CAL--\\n'+'\\n'.join(cal)).encode()).hexdigest(),
  'train_sha256':EXPECTED['train'],'dev_sha256':EXPECTED['dev'], 'pt57_sha256':EXPECTED['pt57'],
  'audio_missing_train':0,'dev_reference_values_read':False,'historical175_used':False,
  'cal56_not_independent_of_original_PT57_training':True}
 (ROOT/'PT102_PREFLIGHT.json').write_text(json.dumps(summary,indent=2))
 print('PT102_PREFLIGHT_PASS: TRAIN225/CAL56, checkpoint SHA, budget, TRAIN audio. DEV values NOT read.',flush=True)
 print('SPLIT_SHA256:',summary['split_identity_sha256'],flush=True)
 return summary
if __name__=='__main__':
 try: check()
 except Exception as e:print('PT102_PREFLIGHT_STOP:',e,flush=True);sys.exit(1)
