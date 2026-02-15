#!/usr/bin/env bash
# Re-run all evaluation scripts (Phase 3–6) from exp_results/generated FASTA
# Usage: from repo root: bash experiments/evaluation/run_all_evaluation.sh
# Optional: SKIP_ESMFOLD=1 to skip ESMFold (structure quality); SKIP_ESMFOLD=1 bash run_all_evaluation.sh

set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
EVAL="$ROOT/experiments/evaluation"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"
export BYPROT_SKIP_DATAMODULES=1
export BYPROT_SKIP_TASKS=1
export HF_HOME="${HF_HOME:-/mnt/hbnas/home/pfp/.cache/huggingface}"

echo "============================================================"
echo "  HIV-1 PR DPLM — Full evaluation (generated -> perplexity/aar/resistance/diversity/mutation/embeddings/report)"
echo "  Working directory: $ROOT"
echo "============================================================"

run() { echo ""; echo ">>> $1"; python3 "$EVAL/$1" || { echo "FAILED: $1"; exit 1; }; }

# No DPLM / lightweight
run "verify_potts_guidance_sign.py"
run "calc_aar.py"
run "resistance_enrichment.py"
run "diversity_analysis.py"
run "mutation_analysis.py"

# Requires DPLM (GPU recommended)
run "calc_perplexity.py"
run "embedding_analysis.py"

# Phase 5: ESMFold (optional; 500 seq/group can be slow)
if [[ "${SKIP_ESMFOLD}" == "1" ]]; then
  echo ""
  echo ">>> Skipping structure_quality.py (ESMFold); SKIP_ESMFOLD=1 set"
else
  run "structure_quality.py"
fi

# Phase 6: Consolidated report and dashboard
run "generate_report.py"

echo ""
echo "============================================================"
echo "  All evaluations done. See: exp_results/dashboard/summary.html"
echo "============================================================"
