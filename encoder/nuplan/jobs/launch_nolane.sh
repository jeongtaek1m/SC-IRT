#!/usr/bin/env bash
set -u
S=$(cd "$(dirname "$0")/.." && pwd)
cd $S
echo "$(date +%T) launching 4 queues (2 per GPU on 2 and 3)"
./worker_nolane.sh 2 < f_nolane_q0.txt >> logs_nolane/q0.txt 2>&1 &
./worker_nolane.sh 2 < f_nolane_q1.txt >> logs_nolane/q1.txt 2>&1 &
./worker_nolane.sh 3 < f_nolane_q2.txt >> logs_nolane/q2.txt 2>&1 &
./worker_nolane.sh 3 < f_nolane_q3.txt >> logs_nolane/q3.txt 2>&1 &
wait
echo "$(date +%T) NOLANE NUPLAN DONE"
