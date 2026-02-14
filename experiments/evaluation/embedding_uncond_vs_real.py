#!/usr/bin/env python3
"""
Uncond vs Real 嵌入可视化
仅对 uncond_gen、naive_real、exper_real 做 PCA/t-SNE。
uncond_gen 使用 1000 条，naive/exper_real 各最多 500 条以平衡可视化。
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

    # 可视化数量：uncond 1000，real 各 500
    max_uncond = 1000
    max_real = 500
    batch_size = 32

    print("=" * 60)
    print("Uncond vs Real 嵌入可视化")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n加载 DPLM...")
    model = DiffusionProteinLanguageModel.from_pretrained("airkingbd/dplm_150m")
    model = model.eval().to(device)
    tokenizer = model.tokenizer

    def get_embeddings(sequences, max_n, desc="embed"):
        seqs = [to_93aa_if_99(s) for s in sequences][:max_n]
        embs = []
        for i in tqdm(range(0, len(seqs), batch_size), desc=desc):
            batch_seqs = seqs[i:i + batch_size]
            batch = tokenizer.batch_encode_plus(
                batch_seqs, add_special_tokens=True, padding='longest', return_tensors='pt'
            )
            input_ids = batch["input_ids"].to(device)
            with torch.no_grad():
                logits, hidden = model(input_ids, return_last_hidden_state=True)
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
    e, n = get_embeddings(naive_seqs, max_real, "naive_real")
    embeddings_list.append(e)
    labels_list.extend(["naive_real"] * n)

    _, exper_seqs = load_sequences(data_dir / "pr_exper_val.fasta")
    e, n = get_embeddings(exper_seqs, max_real, "exper_real")
    embeddings_list.append(e)
    labels_list.extend(["exper_real"] * n)

    _, gu = load_sequences(generated_dir / "pr_uncond.fasta")
    e, n = get_embeddings(gu, max_uncond, "uncond_gen")
    embeddings_list.append(e)
    labels_list.extend(["uncond_gen"] * n)

    X = np.vstack(embeddings_list)
    labels = np.array(labels_list)
    np.savez(output_dir / "embeddings_uncond_vs_real.npz", X=X, labels=labels)

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        color_map = {"naive_real": "C0", "exper_real": "C1", "uncond_gen": "C2"}

        # PCA 50 -> t-SNE
        pca50 = PCA(n_components=50)
        Xpca50 = pca50.fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        Xtsne = tsne.fit_transform(Xpca50)

        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(Xtsne[idx, 0], Xtsne[idx, 1], c=color_map.get(lab, "gray"),
                        label=lab, alpha=0.6, s=10)
        plt.legend(loc="best", fontsize=10)
        plt.title("t-SNE: Uncond (1000) vs Naive/Exper Real")
        plt.tight_layout()
        plt.savefig(output_dir / "tsne_uncond_vs_real.png", dpi=120)
        plt.close()

        # PCA 2D
        pca2 = PCA(n_components=2)
        Xpca2 = pca2.fit_transform(X)
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(Xpca2[idx, 0], Xpca2[idx, 1], c=color_map.get(lab, "gray"),
                        label=lab, alpha=0.6, s=10)
        plt.legend(loc="best", fontsize=10)
        plt.title("PCA: Uncond (1000) vs Naive/Exper Real")
        plt.tight_layout()
        plt.savefig(output_dir / "pca_uncond_vs_real.png", dpi=120)
        plt.close()

        print(f"\n已保存: {output_dir / 'tsne_uncond_vs_real.png'}, {output_dir / 'pca_uncond_vs_real.png'}")
    except Exception as err:
        print(f"\n可视化跳过: {err}")

    print(f"\n保存: {output_dir / 'embeddings_uncond_vs_real.npz'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
