#!/usr/bin/env bash
# One serial worker per GPU for the permutation-fixed null.  Job line: "<tag>_full_s<seed>|<args>".
# Skips a job whose nuPlan npz exists.  Never touches a process it did not start.
set -u
BASE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$BASE/stage2out; LOG=$BASE/logs
PY=/home/jeongtae/miniconda3/envs/smart/bin/python
CD=$BASE/chdrop_perm.py
GPU=$1
while IFS='|' read -r name args; do
  [ -z "$name" ] && continue
  tag=${name%_full_s*}; seed=${name##*_full_s}
  want="$OUT/${tag}_b2d2nuplan_s${seed}.npz"
  if [ -f "$want" ]; then echo "SKIP  $name (have $want)"; continue; fi
  echo "$(date +%H:%M:%S) RUN   $name  gpu $GPU"; t0=$SECONDS
  if $PY "$CD" --gpu "$GPU" --outdir "$OUT" $args > "$LOG/$name.log" 2>&1; then
    echo "$(date +%H:%M:%S) OK    $name  ($((SECONDS-t0))s)"
  else
    echo "$(date +%H:%M:%S) FAIL  $name  ($((SECONDS-t0))s)  see $LOG/$name.log"; tail -5 "$LOG/$name.log"
  fi
done
echo "$(date +%H:%M:%S) WORKER gpu $GPU DONE"
