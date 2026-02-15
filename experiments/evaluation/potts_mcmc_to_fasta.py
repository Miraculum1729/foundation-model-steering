#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Mi3-GPU Potts model MCMC sample sequences to FASTA for DPLM embedding analysis.

Usage:
  python experiments/evaluation/potts_mcmc_to_fasta.py

If Mi3 MCMC sequences are not generated yet, run first:
  bash experiments/evaluation/run_potts_mcmc.sh
"""

import sys
from pathlib import Path


def load_mi3_seqs(seqs_path, max_seqs=500):
    """Load sequences from Mi3 seqs file (one sequence per line, 93 aa)."""
    seqs = []
    with open(seqs_path) as f:
        for i, line in enumerate(f):
            if i >= max_seqs:
                break
            s = line.strip()
            if s and len(s) == 93:
                # Replace gap '-' with 'A' if present
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

    # Naive Potts MCMC sequences
    naive_seqs_path = mi3_root / "pr.naive.splitseq/pr_naive_gen/seqs"
    # Exper Potts MCMC sequences
    exper_seqs_path = mi3_root / "pr.exper.splitseq/pr_exper_gen/seqs"

    max_per_group = 200  # Match embedding_analysis

    for label, path in [("naive", naive_seqs_path), ("exper", exper_seqs_path)]:
        if not path.exists():
            print(f"[Skip] {path} not found. Run Mi3 MCMC first: bash experiments/evaluation/run_potts_mcmc.sh")
            continue
        seqs = load_mi3_seqs(path, max_seqs=max_per_group)
        out_path = out_dir / f"pr_potts_mcmc_{label}.fasta"
        save_fasta(seqs, out_path, prefix=f"POTTS_MCMC_{label.upper()}")
        print(f"Saved {len(seqs)} Potts MCMC ({label}) sequences -> {out_path}")

    print("\nDone. Run embedding analysis: python experiments/evaluation/embedding_analysis.py")


if __name__ == "__main__":
    main()
