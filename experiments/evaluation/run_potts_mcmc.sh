#!/bin/bash
# Mi3-GPU Potts model MCMC sampling
# Sample naive and exper Potts models; output seqs for embedding visualization

set -e
MI3_ROOT="/mnt/hbnas/home/pfp/hiv/Mi3-GPU"
EXAMPLES="$MI3_ROOT/examples"
ALPHA="ACDEFGHIKLMNPQRSTVWY-"

# Check Mi3
if ! command -v Mi3.py &>/dev/null; then
    echo "Please install Mi3-GPU first: cd $MI3_ROOT && pip install -e ."
    exit 1
fi

cd "$MI3_ROOT"

echo "===== Naive Potts MCMC sampling ====="
Mi3.py gen \
  --init_model "$EXAMPLES/pr.naive.splitseq/pr_naive_inference/run_63" \
  --alpha "$ALPHA" \
  --nwalkers 5120 \
  --nsteps 2048 \
  --equiltime 1 \
  --gpus 0:0 \
  --outdir "$EXAMPLES/pr.naive.splitseq/pr_naive_gen"

echo ""
echo "===== Exper Potts MCMC sampling ====="
Mi3.py gen \
  --init_model "$EXAMPLES/pr.exper.splitseq/pr_exper_inference/run_63" \
  --alpha "$ALPHA" \
  --nwalkers 5120 \
  --nsteps 2048 \
  --equiltime 1 \
  --gpus 0:0 \
  --outdir "$EXAMPLES/pr.exper.splitseq/pr_exper_gen"

echo ""
echo "MCMC done. Run: python experiments/evaluation/potts_mcmc_to_fasta.py"
echo "Then: python experiments/evaluation/embedding_analysis.py"
