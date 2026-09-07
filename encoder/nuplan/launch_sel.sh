#!/usr/bin/env bash
set -u
B=$(cd "$(dirname "$0")" && pwd)
cd $B
echo "$(date +%H:%M:%S) smoke on gpu 2"
grep -m1 "^C4r2n_p0_full_s0|" f_sel_gpu2.txt | ./worker_sel.sh 2 >> logs_sel_gpu2.txt 2>&1
/home/jeongtae/miniconda3/envs/smart/bin/python verify_sel_smoke.py || { echo "SMOKE FAILED"; tail -15 logs_sel/C4r2n_p0_full_s0.log; exit 1; }
echo "$(date +%H:%M:%S) launching 2 workers (33 jobs each)"
./worker_sel.sh 2 < f_sel_gpu2.txt >> logs_sel_gpu2.txt 2>&1 &
./worker_sel.sh 3 < f_sel_gpu3.txt >> logs_sel_gpu3.txt 2>&1 &
wait
echo "$(date +%H:%M:%S) ALL DONE"
