#!/usr/bin/env bash
# R2-noLane under the two-stage protocol (--early-stop), with and without --proper-init.
# 1-stage reference: r2nolane_b2d_s{0,1,2}.npz (frozen 30 epochs, no selection).
#   --early-stop               -> es/r2nolane_b2d_s{k}.npz        (init shared across seeds)
#   --early-stop --proper-init -> es_pinit/r2nolane_b2d_s{k}.npz  (init varies with the seed)
set -u
E=/data2/jeongtae/relgraph_e16sel; PY=/home/jeongtae/miniconda3/envs/smart/bin/python
cd $E
for s in 0 1 2; do
  ( $PY r2_graph.py --domain b2d --gpu $s --seed $s --ablate-lane --early-stop \
      > $E/twostage_logs/es_s$s.log 2>&1 && echo "$(date +%T) OK es s$s" || echo "$(date +%T) FAIL es s$s" ) &
  sleep 6
done
wait
for s in 0 1 2; do
  ( $PY r2_graph.py --domain b2d --gpu $s --seed $s --ablate-lane --early-stop --proper-init \
      > $E/twostage_logs/esp_s$s.log 2>&1 && echo "$(date +%T) OK esp s$s" || echo "$(date +%T) FAIL esp s$s" ) &
  sleep 6
done
wait
echo "$(date +%T) TWOSTAGE DONE"
