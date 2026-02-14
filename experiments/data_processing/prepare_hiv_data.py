#!/usr/bin/env python3
"""
HIV-1 PR data preparation script.
Load Stanford PR data and trained Potts models, compute statistics.
"""

import os
import numpy as np
from pathlib import Path


def load_fasta_sequences(filepath):
    """Load FASTA-format sequence file (plain text, one seq per line)"""
    sequences = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                # Remove gap chars, keep standard amino acids
                seq = line.replace('-', '')
                if len(seq) == 93:
                    sequences.append(seq)
    return sequences


def load_potts_model(j_path):
    """
    Load Potts model.

    J matrix shape (4278, 441):
    - L = 93 (sequence length)
    - q = 21 (21 states: 20 amino acids + gap)
    - 4278 = (93 × 92) / 2, upper-triangular site pairs
    - 441 = 21 × 21
    """
    J_full = np.load(j_path)

    L = 93
    q = 21

    print(f"Loading Potts model: {j_path}")
    print(f"J matrix shape: {J_full.shape}")

    # Rebuild full J_{ij}(a,b) tensor [L, L, q, q]
    # J_full row k corresponds to site pair (i,j), k = upper-triangle index
    J_coupling = np.zeros((L, L, q, q))

    k = 0
    for i in range(L):
        for j in range(i+1, L):
            # J_full[k] is (441,), reshape to (21,21)
            J_coupling[i, j, :, :] = J_full[k].reshape(q, q)
            J_coupling[j, i, :, :] = J_full[k].reshape(q, q)  # symmetric
            k += 1

    # External field h may need extraction elsewhere or set to zero
    # Set to zero for now, adjust with real data later
    h = np.zeros((L, q))

    print(f"Rebuilt coupling matrix: {J_coupling.shape}")
    print(f"External field matrix: {h.shape}")

    return {
        'J': J_coupling,
        'h': h,
        'L': L,
        'q': q
    }


def compute_pssm(sequences, num_states=21):
    """Compute position-specific scoring matrix (PSSM)"""
    L = len(sequences[0])
    pssm = np.zeros((L, num_states))

    # Amino acid to index mapping
    aa_to_idx = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }

    for seq in sequences:
        for pos, aa in enumerate(seq):
            if aa in aa_to_idx:
                idx = aa_to_idx[aa]
                pssm[pos, idx] += 1

    # Normalize
    pssm = pssm / len(sequences)

    return pssm


def compute_mutation_frequency(sequences):
    """Compute mutation frequency per site"""
    L = len(sequences[0])
    freq = {}

    for pos in range(L):
        aa_counts = {}
        for seq in sequences:
            aa = seq[pos]
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        freq[pos] = {k: v / len(sequences) for k, v in aa_counts.items()}

    return freq


def compute_shannon_entropy(pssm):
    """Compute Shannon entropy per site"""
    eps = 1e-10  # Avoid log(0)
    entropy = -np.sum(pssm * np.log2(pssm + eps), axis=1)
    return entropy


def extract_hxb2_reference(sequences):
    """Extract sequence closest to consensus as HXB2 reference"""
    # Compute consensus
    consensus_seq = []
    L = len(sequences[0])

    for pos in range(L):
        aa_counts = {}
        for seq in sequences:
            aa = seq[pos]
            aa_counts[aa] = aa_counts.get(aa, 0) + 1

        # Pick most frequent amino acid
        most_common_aa = max(aa_counts.items(), key=lambda x: x[1])[0]
        consensus_seq.append(most_common_aa)

    consensus = ''.join(consensus_seq)

    # Find sequence closest to consensus
    min_dist = float('inf')
    hxb2_seq = None

    for seq in sequences:
        dist = sum(1 for a, b in zip(seq, consensus) if a != b)
        if dist < min_dist:
            min_dist = dist
            hxb2_seq = seq

    return hxb2_seq, consensus


def main():
    # Path config
    data_dir = Path("/mnt/hbnas/home/pfp/hiv/discrete_flow_models/data/hiv_msa")
    potts_dir = Path("/mnt/hbnas/home/pfp/hiv/Mi3-GPU/examples")
    output_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/processed")
    reference_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/reference")

    # Create output dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: HIV-1 PR data preparation")
    print("=" * 60)

    # 1. Load val set (val only for evaluation)
    print("\n[1/5] Loading val set...")

    naive_val_seqs = load_fasta_sequences(data_dir / "pr_naive/val_sequences.txt")
    exper_val_seqs = load_fasta_sequences(data_dir / "pr_exper/val_sequences.txt")

    print(f"  Naive val: {len(naive_val_seqs)} sequences")
    print(f"  Exper val: {len(exper_val_seqs)} sequences")

    # Verify sequence length
    if naive_val_seqs:
        print(f"  Naive seq length: {len(naive_val_seqs[0])} aa")
    if exper_val_seqs:
        print(f"  Exper seq length: {len(exper_val_seqs[0])} aa")

    # 2. Load Potts models
    print("\n[2/5] Loading Potts models...")

    potts_naive_path = potts_dir / "pr.naive.splitseq/pr_naive_inference/run_63/J.npy"
    potts_exper_path = potts_dir / "pr.exper.splitseq/pr_exper_inference/run_63/J.npy"

    potts_naive = load_potts_model(potts_naive_path)
    potts_exper = load_potts_model(potts_exper_path)

    # 3. Compute statistics (from val set)
    print("\n[3/5] Computing statistics...")

    # Naive stats
    print("  Naive stats...")
    naive_pssm = compute_pssm(naive_val_seqs)
    naive_mutation_freq = compute_mutation_frequency(naive_val_seqs)
    naive_entropy = compute_shannon_entropy(naive_pssm)

    print(f"    Mean Shannon entropy: {naive_entropy.mean():.3f}")
    print(f"    Max entropy site: {np.argmax(naive_entropy)} (val: {naive_entropy.max():.3f})")
    print(f"    Min entropy site: {np.argmin(naive_entropy)} (val: {naive_entropy.min():.3f})")

    # Exper stats
    print("  Exper stats...")
    exper_pssm = compute_pssm(exper_val_seqs)
    exper_mutation_freq = compute_mutation_frequency(exper_val_seqs)
    exper_entropy = compute_shannon_entropy(exper_pssm)

    print(f"    Mean Shannon entropy: {exper_entropy.mean():.3f}")
    print(f"    Max entropy site: {np.argmax(exper_entropy)} (val: {exper_entropy.max():.3f})")
    print(f"    Min entropy site: {np.argmin(exper_entropy)} (val: {exper_entropy.min():.3f})")

    # 4. Extract HXB2 reference
    print("\n[4/5] Extracting HXB2 reference...")

    # Use naive val set for HXB2 (naive more conservative)
    hxb2_naive, naive_consensus = extract_hxb2_reference(naive_val_seqs)
    hxb2_exper, exper_consensus = extract_hxb2_reference(exper_val_seqs)

    print(f"  Naive consensus: {naive_consensus}")
    print(f"  Naive HXB2 proxy: {hxb2_naive}")
    print(f"  Exper consensus: {exper_consensus}")
    print(f"  Exper HXB2 proxy: {hxb2_exper}")

    # Use naive HXB2 as main reference
    hxb2_ref = hxb2_naive

    # 5. Save all data
    print("\n[5/5] Saving data...")

    # Save val set sequences
    def save_fasta(seqs, filepath):
        with open(filepath, 'w') as f:
            for i, seq in enumerate(seqs):
                f.write(f">SEQ_{i}\n{seq}\n")

    save_fasta(naive_val_seqs, output_dir / "pr_naive_val.fasta")
    save_fasta(exper_val_seqs, output_dir / "pr_exper_val.fasta")

    # Save HXB2 reference
    with open(reference_dir / "hxb2_pr.fasta", 'w') as f:
        f.write(f">HXB2_PR_Reference\n{hxb2_ref}\n")

    # Save Potts models
    np.savez_compressed(
        output_dir / "potts_models.npz",
        potts_naive_J=potts_naive['J'],
        potts_naive_h=potts_naive['h'],
        potts_exper_J=potts_exper['J'],
        potts_exper_h=potts_exper['h'],
        potts_L=potts_naive['L'],
        potts_q=potts_naive['q']
    )

    # Save statistics
    np.savez_compressed(
        output_dir / "pr_statistics.npz",
        naive_pssm=naive_pssm,
        exper_pssm=exper_pssm,
        naive_entropy=naive_entropy,
        exper_entropy=exper_entropy,
        hxb2_ref=np.array(list(hxb2_ref), dtype='U1')
    )

    print(f"\nSaved!")
    print(f"  Val sequences: {output_dir / 'pr_naive_val.fasta'}, {output_dir / 'pr_exper_val.fasta'}")
    print(f"  HXB2 reference: {reference_dir / 'hxb2_pr.fasta'}")
    print(f"  Potts models: {output_dir / 'potts_models.npz'}")
    print(f"  Statistics: {output_dir / 'pr_statistics.npz'}")

    print("\n" + "=" * 60)
    print("Phase 1 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
