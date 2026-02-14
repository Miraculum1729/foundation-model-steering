#!/bin/bash
# Mi3-GPU Potts 模型 MCMC 采样
# 采样 naive 与 exper 两种 Potts 模型，输出 seqs 供 embedding 可视化

set -e
MI3_ROOT="/mnt/hbnas/home/pfp/hiv/Mi3-GPU"
EXAMPLES="$MI3_ROOT/examples"
ALPHA="ACDEFGHIKLMNPQRSTVWY-"

# 检查 Mi3
if ! command -v Mi3.py &>/dev/null; then
    echo "请先安装 Mi3-GPU: cd $MI3_ROOT && pip install -e ."
    exit 1
fi

cd "$MI3_ROOT"

echo "===== Naive Potts MCMC 采样 ====="
Mi3.py gen \
  --init_model "$EXAMPLES/pr.naive.splitseq/pr_naive_inference/run_63" \
  --alpha "$ALPHA" \
  --nwalkers 5120 \
  --nsteps 2048 \
  --equiltime 1 \
  --gpus 0:0 \
  --outdir "$EXAMPLES/pr.naive.splitseq/pr_naive_gen"

echo ""
echo "===== Exper Potts MCMC 采样 ====="
Mi3.py gen \
  --init_model "$EXAMPLES/pr.exper.splitseq/pr_exper_inference/run_63" \
  --alpha "$ALPHA" \
  --nwalkers 5120 \
  --nsteps 2048 \
  --equiltime 1 \
  --gpus 0:0 \
  --outdir "$EXAMPLES/pr.exper.splitseq/pr_exper_gen"

echo ""
echo "MCMC 完成。运行: python experiments/evaluation/potts_mcmc_to_fasta.py"
echo "然后: python experiments/evaluation/embedding_analysis.py"
