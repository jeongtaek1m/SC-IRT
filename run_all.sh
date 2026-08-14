#!/bin/bash
# Evaluate the released method against the frozen panel, in dependency order.
#
#   bash run_all.sh [outdir] [python]
#
# Step 1 writes the frozen gold anchor that the two encoder steps read, so the order
# is a hard contract rather than a convenience.
#
# The runtime is pinned (CPU, single-threaded) because the published numbers are
# float32 optimisation results and CPU/CUDA reductions disagree in the third
# decimal. See REPRODUCIBILITY.md.
set -euo pipefail

OUT=${1:-expected_local}
PY=${2:-python}
export SCIRT_REPRO=${SCIRT_REPRO:-$(cd "$(dirname "$0")" && pwd)}
cd "$SCIRT_REPRO"
mkdir -p "$OUT"

export SCIRT_DEVICE=${SCIRT_DEVICE:-cpu}
export SCIRT_THREADS=${SCIRT_THREADS:-1}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONPATH="$SCIRT_REPRO:${PYTHONPATH:-}"

run() {
  echo "=== $1 ==="
  # Without pipefail the exit status of the pipeline would be tee's, so a failing
  # experiment would silently leave an empty reference file and the suite would
  # carry on. set -o pipefail is on; the redirect keeps the trap meaningful.
  $PY "experiments/$1.py" > "${OUT}/$1.txt"
}

run gold_anchor        # frozen gold b-hat  (run first)
run noise_ceiling      # panel reliability ceiling 0.904
run encoder_us         # the method's AUROC / MAE / rho row
run encoder_verify     # seed stability + between/within decomposition

echo "ALL DONE -> ${OUT}"
