#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Potts guidance sign: paper assumes logits + lambda*G (favors lower energy).
If guidance is correct, Naive/Exper guided groups should have mean energy < Uncond under the corresponding Potts model;
otherwise the implementation sign may be reversed. Uses numpy only, no torch.
"""
from pathlib import Path
import numpy as np

repo = Path(__file__).resolve().parents[2]
DPLM_LEN = 99
PR_93_START, PR_93_END = 5, 98

# J matrix q=21 corresponds to alpha "-ACDEFGHIKLMNPQRSTVWY"; index 0 = gap, 1..20 = 20 amino acids.
# After slice J[:,:,1:21,1:21] we get (L,L,20,20); 0..19 here = ACDEFGHIKLMNPQRSTVWY (same as Mi3/ALPHA21[1:])
AA_TO_IDX = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 0-19


def compute_potts_energy_np(seq, J):
    """Mi3-GPU gauge-free: E = -sum_{i<j} J_ij(x_i,x_j), coupling J only."""
    L = J.shape[0]
    energy = 0.0
    for i in range(min(L, len(seq))):
        aa_i = seq[i] if i < len(seq) else "A"
        if aa_i not in AA_TO_IDX:
            continue
        ai = AA_TO_IDX[aa_i]
        for j in range(i + 1, min(L, len(seq))):
            aa_j = seq[j] if j < len(seq) else "A"
            if aa_j not in AA_TO_IDX:
                continue
            aj = AA_TO_IDX[aa_j]
            if ai < J.shape[2] and aj < J.shape[3]:
                energy -= J[i, j, ai, aj]
    return energy


def _load(fasta_path):
    names, seqs = [], []
    with open(fasta_path) as f:
        name, seq = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    names.append(name)
                    seqs.append(seq)
                name = line[1:]
                seq = ""
            else:
                seq += line
        if seq:
            names.append(name)
            seqs.append(seq)
    return seqs


def to93(s):
    return s[PR_93_START:PR_93_END] if len(s) == DPLM_LEN else s


def main():
    data_dir = repo / "hiv_data" / "processed"
    gen_dir = repo / "exp_results" / "generated"

    potts = np.load(data_dir / "potts_models.npz")
    J_naive = np.asarray(potts["potts_naive_J"])
    J_exper = np.asarray(potts["potts_exper_J"])
    # Mi3-GPU gauge-free, J only. For q=21 alpha is "-ACDEFGHIKLMNPQRSTVWY"; take [1:21,1:21] for 20 amino acids
    if J_naive.shape[2] == 21:
        J_naive = J_naive[:, :, 1:21, 1:21]
        J_exper = J_exper[:, :, 1:21, 1:21]

    uncond = [to93(s) for s in _load(gen_dir / "pr_uncond.fasta")]
    naive_potts_seqs = [to93(s) for s in _load(gen_dir / "pr_naive_potts.fasta")]
    exper_potts_seqs = [to93(s) for s in _load(gen_dir / "pr_exper_potts.fasta")]

    def mean_energy(seqs, J):
        if not seqs:
            return np.nan
        return np.mean([compute_potts_energy_np(s, J) for s in seqs])

    e_uncond_naive = mean_energy(uncond, J_naive)
    e_naive_naive = mean_energy(naive_potts_seqs, J_naive)
    e_uncond_exper = mean_energy(uncond, J_exper)
    e_exper_exper = mean_energy(exper_potts_seqs, J_exper)

    print("=" * 60)
    print("Potts guidance sign verification (paper: logits + λ·G → should favor lower energy)")
    print("=" * 60)
    print("\nMean energy under Naive Potts model:")
    print(f"  Uncond:        {e_uncond_naive:.2f}")
    print(f"  Naive-Potts:  {e_naive_naive:.2f}")
    print(f"  → Naive-Potts should < Uncond: {'✓ Consistent with paper' if e_naive_naive < e_uncond_naive else '✗ Opposite; implementation sign may be reversed'}")
    print("\nMean energy under Exper Potts model:")
    print(f"  Uncond:        {e_uncond_exper:.2f}")
    print(f"  Exper-Potts:  {e_exper_exper:.2f}")
    print(f"  → Exper-Potts should < Uncond: {'✓ Consistent with paper' if e_exper_exper < e_uncond_exper else '✗ Opposite; implementation sign may be reversed'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
