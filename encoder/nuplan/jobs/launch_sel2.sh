#!/usr/bin/env bash
set -u
B=$(cd "$(dirname "$0")/.." && pwd)
cd $B
echo "$(date +%H:%M:%S) launching 2 workers (30 jobs each: permutations 10-19)"
./worker_sel.sh 2 < f_sel2_gpu2.txt >> logs_sel_gpu2.txt 2>&1 &
./worker_sel.sh 3 < f_sel2_gpu3.txt >> logs_sel_gpu3.txt 2>&1 &
wait
echo "$(date +%H:%M:%S) ALL DONE"
