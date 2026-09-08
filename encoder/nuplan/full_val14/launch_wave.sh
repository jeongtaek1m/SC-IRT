#!/usr/bin/env bash
# 63 stage-2 runs (NLe x3 + C4nl 20 perms x 3 seeds) on the FULL val14 target: 6 workers, 2 per GPU on 0, 2, 3.
set -u
W=/data2/jeongtae/nuplan_full_val14; cd $W/pipeline
echo "$(date +%T) wave start"
./worker_full.sh 2 < jobs_w0.txt > $W/res/wave_w0.log 2>&1 & sleep 3
./worker_full.sh 2 < jobs_w1.txt > $W/res/wave_w1.log 2>&1 & sleep 3
./worker_full.sh 2 < jobs_w2.txt > $W/res/wave_w2.log 2>&1 & sleep 3
./worker_full.sh 3 < jobs_w3.txt > $W/res/wave_w3.log 2>&1 & sleep 3
./worker_full.sh 3 < jobs_w4.txt > $W/res/wave_w4.log 2>&1 & sleep 3
./worker_full.sh 3 < jobs_w5.txt > $W/res/wave_w5.log 2>&1 &
wait
echo "$(date +%T) WAVE DONE: $(ls $W/stage2/*_b2d2nuplan_s*.npz 2>/dev/null | wc -l) nuPlan outputs"
