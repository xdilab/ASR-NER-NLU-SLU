#!/usr/bin/env bash
# Execute via bash; never source into your interactive terminal.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="/data/smgreen1/voxtral_moe_v1/pt108_matched_learned_router_v1_0"
ENV="/data/smgreen1/venvs/voxtral_moe_v1"
RESULT="$HOME/Downloads/PT108_MATCHED_LEARNED_ROUTER_RESULTS_v1_0.zip"
LOG="$OUT/PT108_FULL.log"
mkdir -p "$OUT" "$HOME/Downloads"
if [ -s "$RESULT" ] && unzip -tq "$RESULT" >/dev/null 2>&1; then
  echo "PT108 VALID RESULT ALREADY EXISTS: $RESULT"; exit 0
fi
if [ ! -x "$ENV/bin/python" ]; then echo "PT108 STOP: missing environment $ENV"; exit 1; fi
if ! command -v nvidia-smi >/dev/null 2>&1; then echo 'PT108 STOP: NVIDIA not available'; exit 1; fi
if command -v flock >/dev/null 2>&1; then
  exec 9>"$OUT/.pt108_process.lock"
  if ! flock -n 9; then echo 'PT108 STOP: another run holds the lock; inspect PT108_FULL.log'; exit 1; fi
fi
printf '\n=== PT108 CPU SAFETY AND INPUT PREFLIGHT ===\n'
"$ENV/bin/python" "$ROOT/test_pt108_cpu.py" || exit 1
"$ENV/bin/python" "$ROOT/test_pt108_source.py" || exit 1
"$ENV/bin/python" "$ROOT/pt102_preflight.py" || exit 1
free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ')
if ! [[ "$free0" =~ ^[0-9]+$ ]]; then echo 'PT108 STOP: cannot measure GPU0 free memory'; exit 1; fi
if [ "$free0" -lt 28000 ]; then echo "PT108 STOP: GPU0 free memory $free0 MiB (<28000); will not interrupt another job"; exit 1; fi
printf '\n=== PT108 GPU0: MATCHED TRAINED OMNI-STYLE AND HIERARCHICAL-LOGIT ROUTERS ===\n'
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$ENV/bin/python" "$ROOT/pt108_matched_router_train.py" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then echo "PT108 STOP code=$status; upload $LOG"; exit "$status"; fi
"$ENV/bin/python" - "$OUT" "$RESULT" <<'PYZIP'
from pathlib import Path
import sys,zipfile
out=Path(sys.argv[1]);destination=Path(sys.argv[2])
if not (out/'PT108_RESULT.json').is_file():
    raise RuntimeError('PT108_RESULT.json missing: refuse stale/partial result ZIP')
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='.pt108_process.lock' and p.suffix.lower() not in ('.wav','.flac','.mp3'):
            z.write(p,arcname=str(Path(out.name)/p.relative_to(out)))
with zipfile.ZipFile(destination) as z:
    if z.testzip() is not None:raise RuntimeError('Result ZIP integrity failure')
print('READY_TO_UPLOAD',destination)
PYZIP
