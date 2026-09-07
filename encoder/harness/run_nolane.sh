#!/usr/bin/env bash
# R2-noLane: the lane-free encoder, 3 seeds, 16-draw OOF (same frozen recipe as the default arm).
set -u
E=/data2/jeongtae/relgraph_e16sel; PY=/home/jeongtae/miniconda3/envs/smart/bin/python
cd $E; mkdir -p nolane_logs
for s in 0 1 2; do
  g=$s
  echo "$(date +%T) START nolane seed $s on gpu $g"
  ( $PY r2_graph.py --domain b2d --gpu $g --seed $s --ablate-lane > $E/nolane_logs/nolane_s$s.log 2>&1 \
    && echo "$(date +%T) OK seed $s" || echo "$(date +%T) FAIL seed $s" ) &
  sleep 8
done
wait
echo "$(date +%T) NOLANE DONE"
