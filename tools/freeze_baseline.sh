#!/bin/bash
# Freeze a reproduction oracle by running the ORIGINAL, unmodified B2D scripts
# under the pinned runtime configuration (CPU, single-threaded).
#
# This baseline — not the shipped expected/ — is the correctness oracle for the
# refactor: it isolates the single variable under test (the code), holding the
# machine, the libraries and the device fixed.
#
# usage: bash tools/freeze_baseline.sh <outdir> [python]
set -euo pipefail

OUT=${1:?usage: freeze_baseline.sh <outdir> [python]}
PY=${2:-python}
export SCIRT_REPRO=${SCIRT_REPRO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$SCIRT_REPRO"
mkdir -p "$OUT"

# --- pinned runtime -----------------------------------------------------------
# The scripts choose their device via `cuda if torch.cuda.is_available() else cpu`.
# Hiding the GPUs pins them to CPU without editing a single line of the originals.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0

run() {
  local script=$1 arg=${2:-} suffix=${3:-}
  echo "=== ${script} ==="
  $PY "code/${script}" ${arg} 2>&1 | grep -vE "Warning|warn" > "${OUT}/${script%.py}${suffix}.txt"
}

# Dependency order is a hard contract: b2d_camrisk_boot.py writes the frozen gold
# b-hat that five downstream scripts read.
run b2d_camrisk_boot.py
run b2d_noise_ceiling.py
run b2d_headline_2pl.py "minTTC,risk-field,routegeom,kin+den,smart-ent,agentjepa,bl-cmdkin,GT:" _tab1
run b2d_encoder_us_metrics.py
run b2d_interact_w2a_verify.py
run b2d_hybrid_prereg.py
run b2d_up_dynamic.py
run b2d_ups_dynamic.py

# JSON artifacts are part of the oracle too (compared by value, not by bytes).
cp results/camrisk_boot.json results/headline_2pl.json results/hybrid_prereg.json "${OUT}/" 2>/dev/null || true
echo "FREEZE DONE -> ${OUT}"
