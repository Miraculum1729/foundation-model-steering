#!/usr/bin/env python3
"""
AAR (Amino Acid Retention) calculation script.
Compute retention rate of generated sequences relative to HXB2.
Phase 2: generated seqs 99 aa, 93aa = 99aa[5:98]; HXB2 is 93 aa.
"""

import numpy as np
from pathlib import Path

# Phase 2: DPLM generates 99 aa, 93aa = 99aa[5:98]
DPLM_LEN = 99
PR_93_START, PR_93_END = 5, 98


def to_93aa_if_99(seq):
    """If 99 aa generated seq, take 93aa portion to align with HXB2(93aa)."""
    if len(seq) == DPLM_LEN:
        return seq[PR_93_START:PR_93_END]
    return seq


def load_sequences(fasta_path):
    """Load FASTA sequences"""
    sequences = []
    names = []
    with open(fasta_path, 'r') as f:
        seq = ""
        name = ""
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


def calc_aar(seq, hxb2_seq):
    """
    Compute AAR (Amino Acid Retention).

    Args:
        seq: Sequence to compute (93aa)
        hxb2_seq: HXB2 reference (93aa)

    Returns:
        aar: Retention rate [0, 1]
    """
    if len(seq) != len(hxb2_seq):
        print(f"Warning: sequence length mismatch {len(seq)} vs {len(hxb2_seq)}")
        return np.nan

    matches = sum(1 for a, b in zip(seq, hxb2_seq) if a == b)
    aar = matches / len(seq)

    return aar


def main():
    # Path config
    data_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/processed")
    generated_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/generated")
    output_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/aar")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Phase 4.1: AAR calculation")
    print("="*60)

    # 1. Load HXB2 reference
    print("\n[1/3] Loading HXB2 reference...")

    with open(data_dir / "../reference/hxb2_pr.fasta", 'r') as f:
        for line in f:
            if not line.startswith('>'):
                hxb2_seq = line.strip()
                break

    print(f"HXB2 length: {len(hxb2_seq)} aa")
    print(f"HXB2: {hxb2_seq}")

    # 2. Compute AAR for real sequences
    print("\n[2/3] Computing AAR for real sequences...")

    _, naive_seqs = load_sequences(data_dir / "pr_naive_val.fasta")
    _, exper_seqs = load_sequences(data_dir / "pr_exper_val.fasta")

    print(f"  Naive: {len(naive_seqs)} seqs")
    print(f"  Exper: {len(exper_seqs)} seqs")

    naive_aars = [calc_aar(seq, hxb2_seq) for seq in naive_seqs]
    exper_aars = [calc_aar(seq, hxb2_seq) for seq in exper_seqs]

    print(f"  Naive AAR mean: {np.nanmean(naive_aars)*100:.2f}%")
    print(f"  Exper AAR mean: {np.nanmean(exper_aars)*100:.2f}%")

    # 3. Compute AAR for generated seqs (Phase 2: 99 aa -> take 93aa vs HXB2)
    print("\n[3/3] Computing AAR for generated sequences...")

    _, gen_uncond_seqs = load_sequences(generated_dir / "pr_uncond.fasta")
    _, gen_naive_potts_seqs = load_sequences(generated_dir / "pr_naive_potts.fasta")
    _, gen_exper_potts_seqs = load_sequences(generated_dir / "pr_exper_potts.fasta")

    gen_uncond_seqs = [to_93aa_if_99(s) for s in gen_uncond_seqs]
    gen_naive_potts_seqs = [to_93aa_if_99(s) for s in gen_naive_potts_seqs]
    gen_exper_potts_seqs = [to_93aa_if_99(s) for s in gen_exper_potts_seqs]

    print(f"  Uncond: {len(gen_uncond_seqs)} seqs (93aa)")
    print(f"  Naive-Potts: {len(gen_naive_potts_seqs)} seqs (93aa)")
    print(f"  Exper-Potts: {len(gen_exper_potts_seqs)} seqs (93aa)")

    gen_uncond_aars = [calc_aar(seq, hxb2_seq) for seq in gen_uncond_seqs]
    gen_naive_potts_aars = [calc_aar(seq, hxb2_seq) for seq in gen_naive_potts_seqs]
    gen_exper_potts_aars = [calc_aar(seq, hxb2_seq) for seq in gen_exper_potts_seqs]

    print(f"  Uncond AAR mean: {np.nanmean(gen_uncond_aars)*100:.2f}%")
    print(f"  Naive-Potts AAR mean: {np.nanmean(gen_naive_potts_aars)*100:.2f}%")
    print(f"  Exper-Potts AAR mean: {np.nanmean(gen_exper_potts_aars)*100:.2f}%")

    # 4. Save results
    print("\nSaving results...")

    import csv

    results = []

    for name, aar in zip([f"Naive_{i}" for i in range(len(naive_seqs))], naive_aars):
        results.append({'name': name, 'type': 'naive_real', 'aar': aar})
    for name, aar in zip([f"Exper_{i}" for i in range(len(exper_seqs))], exper_aars):
        results.append({'name': name, 'type': 'exper_real', 'aar': aar})
    for name, aar in zip([f"Uncond_{i}" for i in range(len(gen_uncond_seqs))], gen_uncond_aars):
        results.append({'name': name, 'type': 'uncond_gen', 'aar': aar})
    for name, aar in zip([f"NaivePotts_{i}" for i in range(len(gen_naive_potts_seqs))], gen_naive_potts_aars):
        results.append({'name': name, 'type': 'naive_potts_gen', 'aar': aar})
    for name, aar in zip([f"ExperPotts_{i}" for i in range(len(gen_exper_potts_seqs))], gen_exper_potts_aars):
        results.append({'name': name, 'type': 'exper_potts_gen', 'aar': aar})

    with open(output_dir / "aar_all.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'type', 'aar'])
        writer.writeheader()
        writer.writerows(results)

    # Summary stats
    print("\nSummary stats:")

    summary = {
        'naive_real_mean': np.nanmean(naive_aars),
        'exper_real_mean': np.nanmean(exper_aars),
        'uncond_gen_mean': np.nanmean(gen_uncond_aars),
        'naive_potts_gen_mean': np.nanmean(gen_naive_potts_aars),
        'exper_potts_gen_mean': np.nanmean(gen_exper_potts_aars),
    }

    for key, value in summary.items():
        print(f"  {key}: {value*100:.2f}%")

    # Check if in 40-60% range
    print("\nAAR validation (40-60% range):")

    for key in ['uncond_gen_mean', 'naive_potts_gen_mean', 'exper_potts_gen_mean']:
        value = summary[key]
        in_range = 0.4 <= value <= 0.6
        status = "PASS" if in_range else "FAIL"
        print(f"  {key}: {value*100:.2f}% - {status}")

    with open(output_dir / "aar_summary.txt", 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value:.4f}\n")

    # Viz: AAR distribution (boxplot by type)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        types = [r["type"] for r in results]
        aars = [r["aar"] * 100 for r in results]
        type_order = ["naive_real", "exper_real", "uncond_gen", "naive_potts_gen", "exper_potts_gen"]
        data_by_type = {t: [] for t in type_order}
        for t, a in zip(types, aars):
            if t in data_by_type and not np.isnan(a):
                data_by_type[t].append(a)
        fig, ax = plt.subplots(figsize=(8, 4))
        pos = range(len(type_order))
        ax.boxplot([data_by_type[t] for t in type_order], positions=pos, labels=type_order)
        ax.axhspan(40, 60, alpha=0.15, color="green", label="40-60% target")
        ax.set_ylabel("AAR (%)")
        ax.set_title("AAR distribution by group")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "aar_distribution.png", dpi=120)
        plt.close()
        print(f"  Plot: {output_dir / 'aar_distribution.png'}")
    except Exception as e:
        print("  AAR visualization skipped:", e)

    print(f"\nSaved!")
    print(f"  Detailed: {output_dir / 'aar_all.csv'}")
    print(f"  Summary: {output_dir / 'aar_summary.txt'}")

    print("\n" + "="*60)
    print("Phase 4.1 complete!")
    print("="*60)


if __name__ == "__main__":
    main()
