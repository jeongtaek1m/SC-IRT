#!/usr/bin/env bash
set -u
BASE=$(cd "$(dirname "$0")/.." && pwd)
OUT=/data2/jeongtae/nuplan_full_val14/stage2; LOG=/data2/jeongtae/nuplan_full_val14/logs_stage2; mkdir -p $OUT $LOG
PY=/home/jeongtae/miniconda3/envs/smart/bin/python; CD=/data2/jeongtae/nuplan_full_val14/pipeline/chdrop_nolane_full.py
GPU=$1
while IFS='|' read -r name args; do
  [ -z "$name" ] && continue
  tag=${name%_full_s*}; seed=${name##*_full_s}
  want="$OUT/${tag}_b2d2nuplan_s${seed}.npz"
  if [ -f "$want" ]; then echo "SKIP  $name"; continue; fi
  echo "$(date +%T) RUN   $name gpu $GPU"; t0=$SECONDS
  if $PY "$CD" --gpu "$GPU" --outdir "$OUT" $args > "$LOG/$name.log" 2>&1; then
    echo "$(date +%T) OK    $name ($((SECONDS-t0))s)"
  else
    echo "$(date +%T) FAIL  $name ($((SECONDS-t0))s)"; tail -4 "$LOG/$name.log"
  fi
done
echo "$(date +%T) WORKER gpu $GPU DONE"
