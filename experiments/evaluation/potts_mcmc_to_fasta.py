#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Mi3-GPU Potts 模型 MCMC 采样序列转换为 FASTA，供 DPLM embedding 分析使用。

用法:
  python experiments/evaluation/potts_mcmc_to_fasta.py

若 Mi3 MCMC 序列尚未生成，先运行:
  bash experiments/evaluation/run_potts_mcmc.sh
"""

import sys
from pathlib import Path


def load_mi3_seqs(seqs_path, max_seqs=500):
    """从 Mi3 seqs 文件读取序列（每行一条，93 aa）"""
    seqs = []
    with open(seqs_path) as f:
        for i, line in enumerate(f):
            if i >= max_seqs:
                break
            s = line.strip()
            if s and len(s) == 93:
                # 将 gap '-' 替换为 'A'（若存在）
                s = s.replace('-', 'A')
                seqs.append(s)
    return seqs


def save_fasta(seqs, out_path, prefix="SEQ"):
    with open(out_path, 'w') as f:
        for i, seq in enumerate(seqs):
            f.write(f">{prefix}_{i}\n{seq}\n")


def main():
    mi3_root = Path("/mnt/hbnas/home/pfp/hiv/Mi3-GPU/examples")
    out_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Naive Potts MCMC 序列
    naive_seqs_path = mi3_root / "pr.naive.splitseq/pr_naive_gen/seqs"
    # Exper Potts MCMC 序列
    exper_seqs_path = mi3_root / "pr.exper.splitseq/pr_exper_gen/seqs"

    max_per_group = 200  # 与 embedding_analysis 一致

    for label, path in [("naive", naive_seqs_path), ("exper", exper_seqs_path)]:
        if not path.exists():
            print(f"[跳过] {path} 不存在。请先运行 Mi3 MCMC: bash experiments/evaluation/run_potts_mcmc.sh")
            continue
        seqs = load_mi3_seqs(path, max_seqs=max_per_group)
        out_path = out_dir / f"pr_potts_mcmc_{label}.fasta"
        save_fasta(seqs, out_path, prefix=f"POTTS_MCMC_{label.upper()}")
        print(f"已保存 {len(seqs)} 条 Potts MCMC ({label}) 序列 -> {out_path}")

    print("\n完成。可运行 embedding 分析: python experiments/evaluation/embedding_analysis.py")


if __name__ == "__main__":
    main()
