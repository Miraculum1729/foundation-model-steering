#!/usr/bin/env python3
"""
Phase 3.3: Mutation distribution comparison.
Single-site mutation frequency, PSSM, key-site KL divergence. 93aa = 99aa[5:98].
"""

import numpy as np
from pathlib import Path
from collections import Counter

try:
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DPLM_LEN = 99
PR_93_START, PR_93_END = 5, 98
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA)}
KEY_SITES_1BASED = [30, 46, 63, 71, 82, 84, 90]  # PR key sites


def to_93aa_if_99(seq):
    if len(seq) == DPLM_LEN:
        return seq[PR_93_START:PR_93_END]
    return seq


def load_sequences(fasta_path):
    sequences, names = [], []
    with open(fasta_path, 'r') as f:
        seq, name = "", ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    names.append(name)
                name = line[1:]
                seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq)
            names.append(name)
    return names, sequences


def pssm_from_sequences(sequences, L=93):
    """PSSM: per-site 20-letter frequency, shape (L, 20)."""
    cnt = np.zeros((L, 20))
    for s in sequences:
        s93 = to_93aa_if_99(s) if len(s) == DPLM_LEN else s
        if len(s93) != L:
            continue
        for i, a in enumerate(s93):
            if a in AA_TO_IDX:
                cnt[i, AA_TO_IDX[a]] += 1
    total = cnt.sum(axis=1, keepdims=True)
    total[total == 0] = 1
    return cnt / total


def kl_divergence_per_position(p_real, p_gen, eps=1e-10):
    """Per-site KL(real || gen), returns (L,) or scalar."""
    if not HAS_SCIPY:
        return None
    p_r = np.clip(p_real, eps, 1)
    p_g = np.clip(p_gen, eps, 1)
    return scipy.stats.entropy(p_r.T, p_g.T, axis=0)


def main():
    data_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/processed")
    generated_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/generated")
    output_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/mutation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3.3: Mutation distribution comparison")
    print("=" * 60)

    _, naive_seqs = load_sequences(data_dir / "pr_naive_val.fasta")
    _, exper_seqs = load_sequences(data_dir / "pr_exper_val.fasta")
    _, gu = load_sequences(generated_dir / "pr_uncond.fasta")
    _, gn = load_sequences(generated_dir / "pr_naive_potts.fasta")
    _, ge = load_sequences(generated_dir / "pr_exper_potts.fasta")
    gu = [to_93aa_if_99(s) for s in gu]
    gn = [to_93aa_if_99(s) for s in gn]
    ge = [to_93aa_if_99(s) for s in ge]

    L = 93
    pssm_naive = pssm_from_sequences(naive_seqs, L)
    pssm_exper = pssm_from_sequences(exper_seqs, L)
    pssm_uncond = pssm_from_sequences(gu, L)
    pssm_naive_potts = pssm_from_sequences(gn, L)
    pssm_exper_potts = pssm_from_sequences(ge, L)

    # PSSM Frobenius similarity (1 - norm_diff / max_norm)
    def frob_sim(a, b):
        d = np.linalg.norm(a - b, 'fro')
        m = max(np.linalg.norm(a, 'fro'), np.linalg.norm(b, 'fro'), 1e-10)
        return max(0, 1 - d / m)

    rows = [
        ("naive_real", "exper_real", frob_sim(pssm_naive, pssm_exper)),
        ("naive_real", "uncond_gen", frob_sim(pssm_naive, pssm_uncond)),
        ("naive_real", "naive_potts_gen", frob_sim(pssm_naive, pssm_naive_potts)),
        ("naive_real", "exper_potts_gen", frob_sim(pssm_naive, pssm_exper_potts)),
        ("exper_real", "uncond_gen", frob_sim(pssm_exper, pssm_uncond)),
        ("exper_real", "naive_potts_gen", frob_sim(pssm_exper, pssm_naive_potts)),
        ("exper_real", "exper_potts_gen", frob_sim(pssm_exper, pssm_exper_potts)),
    ]
    with open(output_dir / "pssm_similarity.csv", "w") as f:
        f.write("group_a,group_b,pssm_similarity\n")
        for a, b, s in rows:
            f.write(f"{a},{b},{s:.6f}\n")
    print("\nPSSM similarity (Frobenius):")
    for a, b, s in rows:
        print(f"  {a} vs {b}: {s:.4f}")

    # Per-site KL(exper || gen) and KL(naive || gen)
    if HAS_SCIPY:
        kl_exper_uncond = kl_divergence_per_position(pssm_exper, pssm_uncond)
        kl_exper_naive_potts = kl_divergence_per_position(pssm_exper, pssm_naive_potts)
        kl_exper_exper_potts = kl_divergence_per_position(pssm_exper, pssm_exper_potts)
        with open(output_dir / "kl_divergence_by_pos.csv", "w") as f:
            f.write("position_1based,kl_exper_vs_uncond,kl_exper_vs_naive_potts,kl_exper_vs_exper_potts\n")
            for i in range(L):
                pos1 = i + 1
                f.write(f"{pos1},{kl_exper_uncond[i]:.6f},{kl_exper_naive_potts[i]:.6f},{kl_exper_exper_potts[i]:.6f}\n")
        print("\nKey-site KL(exper || gen):")
        for pos1 in KEY_SITES_1BASED:
            i = pos1 - 1
            if i < L:
                print(f"  Site{pos1}: uncond={kl_exper_uncond[i]:.4f}, naive_potts={kl_exper_naive_potts[i]:.4f}, exper_potts={kl_exper_exper_potts[i]:.4f}")

        # Viz: key-site KL comparison
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            x = np.arange(len(KEY_SITES_1BASED))
            w = 0.25
            u = [kl_exper_uncond[s - 1] for s in KEY_SITES_1BASED if s <= L]
            n = [kl_exper_naive_potts[s - 1] for s in KEY_SITES_1BASED if s <= L]
            e = [kl_exper_exper_potts[s - 1] for s in KEY_SITES_1BASED if s <= L]
            plt.bar(x - w, u, w, label="uncond_gen", color="C0")
            plt.bar(x, n, w, label="naive_potts_gen", color="C1")
            plt.bar(x + w, e, w, label="exper_potts_gen", color="C2")
            plt.xticks(x, [str(s) for s in KEY_SITES_1BASED])
            plt.xlabel("Key site (1-based PR)")
            plt.ylabel("KL(exper_real || gen)")
            plt.legend()
            plt.title("Mutation: KL divergence at key sites")
            plt.tight_layout()
            plt.savefig(output_dir / "key_sites_comparison.png", dpi=120)
            plt.close()
            print("  Plot: key_sites_comparison.png")
        except Exception as err:
            print("  Visualization skipped:", err)
    else:
        print("\n(scipy not installed, skipping KL divergence)")

    print(f"\nSaved: {output_dir}")
    print("  pssm_similarity.csv, kl_divergence_by_pos.csv")
    print("=" * 60)
    print("Phase 3.3 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
