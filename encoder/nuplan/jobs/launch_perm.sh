#!/usr/bin/env bash
set -u
B=$(cd "$(dirname "$0")/.." && pwd)
cd $B
echo "$(date +%H:%M:%S) smoke on gpu 2"
head -1 f_perm_gpu2.txt | ./worker_perm.sh 2 >> logs_perm_gpu2.txt 2>&1
/home/jeongtae/miniconda3/envs/smart/bin/python verify_smoke.py || { echo "SMOKE FAILED"; exit 1; }
echo "$(date +%H:%M:%S) launching 2 workers"
./worker_perm.sh 2 < f_perm_gpu2.txt >> logs_perm_gpu2.txt 2>&1 &
./worker_perm.sh 3 < f_perm_gpu3.txt >> logs_perm_gpu3.txt 2>&1 &
wait
echo "$(date +%H:%M:%S) ALL DONE"
