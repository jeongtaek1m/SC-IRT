#!/usr/bin/env bash
set -u
S=$(cd "$(dirname "$0")/.." && pwd)
cd $S
echo "$(date +%T) launching 6 queues (3 per GPU on 2 and 3)"
for q in 0 1 2; do ./worker_nla.sh 2 < f_nla_q$q.txt >> logs_nla/q$q.txt 2>&1 & sleep 3; done
for q in 3 4 5; do ./worker_nla.sh 3 < f_nla_q$q.txt >> logs_nla/q$q.txt 2>&1 & sleep 3; done
wait
echo "$(date +%T) NLA NUPLAN DONE"
