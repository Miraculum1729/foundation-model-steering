#!/usr/bin/env python3
"""
Phase 3.2: Embedding extraction and visualization.
Extract embeddings via DPLM forward(..., return_last_hidden_state=True),
reduce with PCA/t-SNE and plot. 93aa sequences consistent with Phase 2.
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/mnt/hbnas/home/pfp/hiv2026/dplm/src')
os.environ["BYPROT_SKIP_DATAMODULES"] = "1"
os.environ["BYPROT_SKIP_TASKS"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/hbnas/home/pfp/.cache/huggingface")

DPLM_LEN = 99
PR_93_START, PR_93_END = 5, 98


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


def main():
    import torch
    from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
    from tqdm import tqdm

    data_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/processed")
    generated_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/generated")
    output_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3.2: Embedding extraction and visualization")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nLoading DPLM...")
    model = DiffusionProteinLanguageModel.from_pretrained("airkingbd/dplm_150m")
    model = model.eval().to(device)
    tokenizer = model.tokenizer

    # Max 200 per group to control time and memory
    max_per_group = 200
    batch_size = 32

    def get_embeddings(sequences, desc="embed"):
        seqs = [to_93aa_if_99(s) for s in sequences][:max_per_group]
        embs = []
        for i in tqdm(range(0, len(seqs), batch_size), desc=desc):
            batch_seqs = seqs[i:i + batch_size]
            batch = tokenizer.batch_encode_plus(
                batch_seqs, add_special_tokens=True, padding='longest', return_tensors='pt'
            )
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                logits, hidden = model(input_ids, return_last_hidden_state=True)
            # hidden: (B, L, D), mean over L -> (B, D)
            mask = (input_ids != model.pad_id) & (input_ids != model.bos_id) & (input_ids != model.eos_id)
            mask = mask.float()
            if mask.sum(1).min() == 0:
                h = hidden.mean(1)
            else:
                h = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            embs.append(h.cpu().numpy())
        return np.vstack(embs), len(seqs)

    labels_list = []
    embeddings_list = []

    _, naive_seqs = load_sequences(data_dir / "pr_naive_val.fasta")
    e, n = get_embeddings(naive_seqs, "naive_real")
    embeddings_list.append(e)
    labels_list.extend(["naive_real"] * n)

    _, exper_seqs = load_sequences(data_dir / "pr_exper_val.fasta")
    e, n = get_embeddings(exper_seqs, "exper_real")
    embeddings_list.append(e)
    labels_list.extend(["exper_real"] * n)

    _, gu = load_sequences(generated_dir / "pr_uncond.fasta")
    e, n = get_embeddings(gu, "uncond_gen")
    embeddings_list.append(e)
    labels_list.extend(["uncond_gen"] * n)

    _, gn = load_sequences(generated_dir / "pr_naive_potts.fasta")
    e, n = get_embeddings(gn, "naive_potts_gen")
    embeddings_list.append(e)
    labels_list.extend(["naive_potts_gen"] * n)

    _, ge = load_sequences(generated_dir / "pr_exper_potts.fasta")
    e, n = get_embeddings(ge, "exper_potts_gen")
    embeddings_list.append(e)
    labels_list.extend(["exper_potts_gen"] * n)

    # Potts model pure MCMC samples (Mi3-GPU gen)
    for label, fname in [("naive", "pr_potts_mcmc_naive.fasta"), ("exper", "pr_potts_mcmc_exper.fasta")]:
        fpath = generated_dir / fname
        if fpath.exists():
            _, g = load_sequences(fpath)
            e, n = get_embeddings(g, f"potts_mcmc_{label}")
            embeddings_list.append(e)
            labels_list.extend([f"potts_mcmc_{label}"] * n)

    X = np.vstack(embeddings_list)
    labels = np.array(labels_list)
    np.savez(output_dir / "embeddings.npz", X=X, labels=labels)

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # PCA 50
        pca50 = PCA(n_components=50)
        Xpca50 = pca50.fit_transform(X)
        # t-SNE on 50-dim
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        Xtsne = tsne.fit_transform(Xpca50)

        color_map = {"naive_real": "C0", "exper_real": "C1", "uncond_gen": "C2",
                     "naive_potts_gen": "C3", "exper_potts_gen": "C4",
                     "potts_mcmc_naive": "C5", "potts_mcmc_exper": "C6"}
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(Xtsne[idx, 0], Xtsne[idx, 1], c=color_map.get(lab, "gray"),
                        label=lab, alpha=0.6, s=10)
        plt.legend(loc="best", fontsize=8)
        plt.title("t-SNE of DPLM embeddings (Phase 3.2)")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_plot.png", dpi=120)
        plt.close()

        # New: t-SNE with exper and uncond only
        for lab in ["exper_real", "uncond_gen"]:
            idx = labels == lab
            if idx.any():
                plt.scatter(Xtsne[idx, 0], Xtsne[idx, 1], c=color_map.get(lab, "gray"),
                            label=lab, alpha=0.6, s=15)
        plt.legend(loc="best", fontsize=10)
        plt.title("t-SNE: exper_real vs uncond_gen")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_exper_uncond.png", dpi=120)
        plt.close()

        # New: t-SNE with naive and uncond only
        for lab in ["naive_real", "uncond_gen"]:
            idx = labels == lab
            if idx.any():
                plt.scatter(Xtsne[idx, 0], Xtsne[idx, 1], c=color_map.get(lab, "gray"),
                            label=lab, alpha=0.6, s=15)
        plt.legend(loc="best", fontsize=10)
        plt.title("t-SNE: naive_real vs uncond_gen")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_naive_uncond.png", dpi=120)
        plt.close()

        # New: exper_real + uncond + potts_mcmc_exper + exper_potts_gen
        for lab in ["exper_real", "uncond_gen", "potts_mcmc_exper", "exper_potts_gen"]:
            idx = labels == lab
            if idx.any():
                plt.scatter(Xtsne[idx, 0], Xtsne[idx, 1], c=color_map.get(lab, "gray"),
                            label=lab, alpha=0.6, s=12)
        plt.legend(loc="best", fontsize=9)
        plt.title("t-SNE: exper_real, uncond_gen, potts_mcmc_exper, exper_potts_gen")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_exper_full.png", dpi=120)
        plt.close()

        # New: naive_real + uncond + potts_mcmc_naive + naive_potts_gen
        for lab in ["naive_real", "uncond_gen", "potts_mcmc_naive", "naive_potts_gen"]:
            idx = labels == lab
            if idx.any():
                plt.scatter(Xtsne[idx, 0], Xtsne[idx, 1], c=color_map.get(lab, "gray"),
                            label=lab, alpha=0.6, s=12)
        plt.legend(loc="best", fontsize=9)
        plt.title("t-SNE: naive_real, uncond_gen, potts_mcmc_naive, naive_potts_gen")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_naive_full.png", dpi=120)
        plt.close()

        # PCA 2D
        pca2 = PCA(n_components=2)
        Xpca2 = pca2.fit_transform(X)
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(Xpca2[idx, 0], Xpca2[idx, 1], c=color_map.get(lab, "gray"),
                        label=lab, alpha=0.6, s=10)
        plt.legend(loc="best", fontsize=8)
        plt.title("PCA of DPLM embeddings (Phase 3.2)")
        plt.tight_layout()
        plt.savefig(output_dir / "pca_plot.png", dpi=120)
        plt.close()
        print(f"\nSaved: tsne_plot, tsne_exper_uncond, tsne_naive_uncond, "
              f"tsne_exper_full, tsne_naive_full, pca_plot")
    except Exception as err:
        print(f"\nVisualization skipped: {err}")

    print(f"\nSaved: {output_dir / 'embeddings.npz'}")
    print("=" * 60)
    print("Phase 3.2 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
