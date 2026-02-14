#!/usr/bin/env python3
"""
Perplexity calculation script.
Compute perplexity for real and generated sequences.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/mnt/hbnas/home/pfp/hiv2026/dplm/src')

os.environ["BYPROT_SKIP_DATAMODULES"] = "1"
os.environ["BYPROT_SKIP_TASKS"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/hbnas/home/pfp/.cache/huggingface")

from byprot.models.dplm.dplm import DiffusionProteinLanguageModel

# Phase 2: DPLM generates 99 aa, 93aa = 99aa[5:98]
DPLM_LEN = 99
PR_93_START = 5
PR_93_END = 98   # 93aa = seq[5:98]


def to_93aa_if_99(seq):
    """If seq is 99 aa (Phase 2), take 93aa portion to align with real data."""
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


def calc_perplexity_batch(model, sequences, tokenizer, device, batch_size=32):
    """Batch compute perplexity"""
    perplexities = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Computing Perplexity"):
        batch_seqs = sequences[i:i+batch_size]

        batch = tokenizer.batch_encode_plus(
            batch_seqs,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
        )

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            logits = model(batch["input_ids"])
            pred_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Compute perplexity per sequence
        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            input_ids = tokenizer.encode(seq, add_special_tokens=True)
            labels = input_ids[1:-1]  # Remove CLS and EOS

            # Create mask (exclude special tokens)
            mask = np.ones(seq_len)
            mask[0] = 0  # CLS
            mask[-1] = 0  # EOS

            if len(labels) > 0:
                # Get logits at positions corresponding to seq length
                seq_logits = pred_probs[j, 1:seq_len+1, :]
                seq_probs = seq_logits[np.arange(len(labels))[:, None], labels]
                
                ce = -np.log(seq_probs + 1e-10)
                perplexity = np.exp(np.mean(ce))
                perplexities.append(perplexity)
            else:
                perplexities.append(np.nan)

    return perplexities


def main():
    # Path config
    data_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/processed")
    generated_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/generated")
    output_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/perplexity")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Phase 3.1: Perplexity calculation")
    print("="*60)

    # Load DPLM model
    print("\nLoading DPLM model...")
    model = DiffusionProteinLanguageModel.from_pretrained("airkingbd/dplm_150m")
    tokenizer = model.tokenizer
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 1. Perplexity for real sequences
    print("\n[1/5] Computing perplexity for real sequences...")

    naive_names, naive_seqs = load_sequences(data_dir / "pr_naive_val.fasta")
    exper_names, exper_seqs = load_sequences(data_dir / "pr_exper_val.fasta")

    print(f"  Naive: {len(naive_seqs)} seqs")
    print(f"  Exper: {len(exper_seqs)} seqs")

    naive_perplexities = calc_perplexity_batch(model, naive_seqs, tokenizer, device)
    exper_perplexities = calc_perplexity_batch(model, exper_seqs, tokenizer, device)

    # 2. Perplexity for generated seqs (Phase 2 outputs 99 aa, take 93aa for comparison)
    print("\n[2/5] Computing perplexity for generated sequences...")

    gen_uncond_names, gen_uncond_seqs = load_sequences(generated_dir / "pr_uncond.fasta")
    gen_naive_potts_names, gen_naive_potts_seqs = load_sequences(generated_dir / "pr_naive_potts.fasta")
    gen_exper_potts_names, gen_exper_potts_seqs = load_sequences(generated_dir / "pr_exper_potts.fasta")

    gen_uncond_seqs = [to_93aa_if_99(s) for s in gen_uncond_seqs]
    gen_naive_potts_seqs = [to_93aa_if_99(s) for s in gen_naive_potts_seqs]
    gen_exper_potts_seqs = [to_93aa_if_99(s) for s in gen_exper_potts_seqs]

    print(f"  Uncond: {len(gen_uncond_seqs)} seqs (93aa)")
    print(f"  Naive-Potts: {len(gen_naive_potts_seqs)} seqs (93aa)")
    print(f"  Exper-Potts: {len(gen_exper_potts_seqs)} seqs (93aa)")

    gen_uncond_perplexities = calc_perplexity_batch(model, gen_uncond_seqs, tokenizer, device)
    gen_naive_potts_perplexities = calc_perplexity_batch(model, gen_naive_potts_seqs, tokenizer, device)
    gen_exper_potts_perplexities = calc_perplexity_batch(model, gen_exper_potts_seqs, tokenizer, device)

    # 3. Save results
    print("\n[3/5] Saving results...")

    import csv

    results = []
    for name, ppl in zip(naive_names, naive_perplexities):
        results.append({'name': name, 'type': 'naive_real', 'perplexity': ppl})
    for name, ppl in zip(exper_names, exper_perplexities):
        results.append({'name': name, 'type': 'exper_real', 'perplexity': ppl})
    for name, ppl in zip(gen_uncond_names, gen_uncond_perplexities):
        results.append({'name': name, 'type': 'uncond_gen', 'perplexity': ppl})
    for name, ppl in zip(gen_naive_potts_names, gen_naive_potts_perplexities):
        results.append({'name': name, 'type': 'naive_potts_gen', 'perplexity': ppl})
    for name, ppl in zip(gen_exper_potts_names, gen_exper_potts_perplexities):
        results.append({'name': name, 'type': 'exper_potts_gen', 'perplexity': ppl})

    with open(output_dir / "perplexity_all.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'type', 'perplexity'])
        writer.writeheader()
        writer.writerows(results)

    # Summary stats
    print("\n[4/5] Summary stats...")

    summary = {
        'naive_real_mean': np.nanmean(naive_perplexities),
        'exper_real_mean': np.nanmean(exper_perplexities),
        'uncond_gen_mean': np.nanmean(gen_uncond_perplexities),
        'naive_potts_gen_mean': np.nanmean(gen_naive_potts_perplexities),
        'exper_potts_gen_mean': np.nanmean(gen_exper_potts_perplexities),
    }

    for key, value in summary.items():
        print(f"  {key}: {value:.3f}")

    with open(output_dir / "perplexity_summary.txt", 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value:.3f}\n")

    print(f"\nSaved!")
    print(f"  Detailed: {output_dir / 'perplexity_all.csv'}")
    print(f"  Summary: {output_dir / 'perplexity_summary.txt'}")

    print("\n" + "="*60)
    print("Phase 3.1 complete!")
    print("="*60)


if __name__ == "__main__":
    main()
