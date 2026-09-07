#!/usr/bin/env bash
# R2-noLane + ego-speed removed: the lane-free counterpart of the UPS no-speed prior.
set -u
E=/data2/jeongtae/relgraph_e16sel; PY=/home/jeongtae/miniconda3/envs/smart/bin/python
cd $E
for s in 0 1 2; do
  g=$s
  echo "$(date +%T) START nolane+nospeed seed $s gpu $g"
  ( $PY drop_channel_nolane.py --drop speed --seed $s --gpu $g \
       --outdir $E/nolane_nospeed --tag nlnospeed > $E/nolane_logs/nlnospeed_s$s.log 2>&1 \
    && echo "$(date +%T) OK seed $s" || echo "$(date +%T) FAIL seed $s" ) &
  sleep 8
done
wait
echo "$(date +%T) NOLANE-NOSPEED DONE"
