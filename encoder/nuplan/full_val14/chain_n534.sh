#!/usr/bin/env bash
# steps 2-5 of the 534-scenario tensor pipeline; waits for step 1 (centerlines) by FILE.
set -u
W=/data2/jeongtae/nuplan_full_val14; P=/home/jeongtae/miniconda3/envs/smart/bin/python
say(){ echo "$(date +%H:%M:%S) $*"; }
until [ -f /data1/jeongtae/smart_difficulty/routes/val14n534_centerlines.pkl ] && grep -qE "wrote|WROTE|done|skips" $W/res/step1_centerlines.log; do
  if grep -qE "Traceback" $W/res/step1_centerlines.log; then say "STEP1 FAILED"; exit 1; fi; sleep 30; done
say "step 1 done: $($P -c "import pickle;print(len(pickle.load(open('/data1/jeongtae/smart_difficulty/routes/val14n534_centerlines.pkl','rb'))))") centerlines"
say "step 2: routed pkls"; $P $W/pipeline/make_routed_pkls_n534.py --split val14n534 > $W/res/step2_routed.log 2>&1 && say "step 2 OK ($(ls /data1/jeongtae/smart_difficulty/pkls/routed_pkls_val14n534 | wc -l) pkls)" || { say "STEP2 FAILED"; exit 1; }
say "step 3: interaction tensors"; $P $W/pipeline/interact_data_n534.py > $W/res/step3_tensors.log 2>&1 && say "step 3 OK" || { say "STEP3 FAILED"; exit 1; }
say "step 4: logged ego"; $P $W/pipeline/build_ego_logged_n534.py > $W/res/step4_ego.log 2>&1 && say "step 4 OK" || { say "STEP4 FAILED"; exit 1; }
say "step 5: relgraph extraction"; $P $W/pipeline/nuplan_extract_n534.py > $W/res/step5_extract.log 2>&1 && say "step 5 OK" || { say "STEP5 FAILED"; exit 1; }
say "PIPELINE DONE"
