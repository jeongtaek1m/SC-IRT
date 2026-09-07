#!/usr/bin/env bash
# Reproduction of the shipped R2 OOF encoder runs (identical recipe, same seeds).
# Writes r2_b2d_s{k}.npz in place; the shipped copies are in repro_backup/.
set -u
E=/data2/jeongtae/relgraph_e16sel; PY=/home/jeongtae/miniconda3/envs/smart/bin/python
cd $E
for s in 0 1 2; do
  g=$(( 2 + s % 2 ))
  echo "$(date +%T) START seed $s on gpu $g"
  ( $PY r2_graph.py --domain b2d --gpu $g --seed $s > $E/repro_backup/repro_s$s.log 2>&1 \
    && echo "$(date +%T) OK seed $s" || echo "$(date +%T) FAIL seed $s" ) &
  sleep 10
done
wait
echo "$(date +%T) REPRO DONE"
