#!/usr/bin/env python3
"""
使用 ESM 对同一批生成序列做 embedding 并画 t-SNE
与 embedding_analysis.py 结构对应，但用 ESM 替代 DPLM 提取表示。

依赖（venv_esmfold 环境已有 esm，需额外）:
  pip install scikit-learn matplotlib

用法:
  source venv_esmfold/bin/activate
  python experiments/evaluation/esm_embedding_tsne.py

或从仓库根目录:
  venv_esmfold/bin/python experiments/evaluation/esm_embedding_tsne.py

输出:
  exp_results/embeddings/embeddings_esm.npz
  exp_results/embeddings/tsne_esm.png
  exp_results/embeddings/tsne_esm_exper_uncond.png
  exp_results/embeddings/tsne_esm_naive_uncond.png
  exp_results/embeddings/pca_esm.png
"""

import os
import sys
import numpy as np
from pathlib import Path

# 路径配置（与 embedding_analysis.py 一致）
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "hiv_data" / "processed"
GENERATED_DIR = ROOT / "exp_results" / "generated"
OUTPUT_DIR = ROOT / "exp_results" / "embeddings"

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


def get_esm_embeddings(model, batch_converter, sequences, device, max_per_group=200, batch_size=16, desc=""):
    """用 ESM 提取 mean-pooled embedding，排除 BOS/EOS/PAD"""
    import torch
    seqs = [to_93aa_if_99(s) for s in sequences][:max_per_group]
    from tqdm import tqdm
    embs = []
    for i in tqdm(range(0, len(seqs), batch_size), desc=desc or "embed", leave=False):
        batch_seqs = seqs[i:i + batch_size]
        batch_labels = [(f"seq_{j}", s) for j, s in enumerate(batch_seqs)]
        _, _, toks = batch_converter(batch_labels)
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[33])
        rep = out["representations"][33]  # (B, L, D)
        # ESM special: BOS=0, PAD=1, EOS=2
        mask = (toks != 0) & (toks != 1) & (toks != 2)
        mask = mask.float()
        if mask.sum(1).min() == 0:
            h = rep.mean(1)
        else:
            h = (rep * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        embs.append(h.cpu().numpy())
    return np.vstack(embs), len(seqs)


def main():
    import torch
    import esm

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ESM 嵌入空间提取与 t-SNE 可视化")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n加载 ESM-2 650M，设备: {device}")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    max_per_group = 200
    labels_list = []
    embeddings_list = []

    def add_group(label, fasta_path):
        if not fasta_path.exists():
            print(f"  跳过（不存在）: {fasta_path}")
            return
        _, seqs = load_sequences(fasta_path)
        e, n = get_esm_embeddings(model, batch_converter, seqs, device, max_per_group, batch_size=16, desc=label)
        embeddings_list.append(e)
        labels_list.extend([label] * n)
        print(f"  {label}: {n} 条")

    # 与 embedding_analysis.py 相同的组
    print("\n加载序列并提取 ESM embedding...")
    add_group("naive_real", DATA_DIR / "pr_naive_val.fasta")
    add_group("exper_real", DATA_DIR / "pr_exper_val.fasta")
    add_group("uncond_gen", GENERATED_DIR / "pr_uncond.fasta")
    add_group("naive_potts_gen", GENERATED_DIR / "pr_naive_potts.fasta")
    add_group("exper_potts_gen", GENERATED_DIR / "pr_exper_potts.fasta")
    add_group("potts_mcmc_naive", GENERATED_DIR / "pr_potts_mcmc_naive.fasta")
    add_group("potts_mcmc_exper", GENERATED_DIR / "pr_potts_mcmc_exper.fasta")

    if not embeddings_list:
        print("错误: 未找到任何序列")
        sys.exit(1)

    X = np.vstack(embeddings_list)
    labels = np.array(labels_list)
    out_npz = OUTPUT_DIR / "embeddings_esm.npz"
    np.savez(out_npz, X=X, labels=labels)
    print(f"\n保存: {out_npz}")

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # PCA 50 -> t-SNE 2
        pca50 = PCA(n_components=50)
        Xpca50 = pca50.fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        Xtsne = tsne.fit_transform(Xpca50)

        color_map = {
            "naive_real": "C0", "exper_real": "C1", "uncond_gen": "C2",
            "naive_potts_gen": "C3", "exper_potts_gen": "C4",
            "potts_mcmc_naive": "C5", "potts_mcmc_exper": "C6",
        }

        # 全量 t-SNE
        fig, ax = plt.subplots(figsize=(8, 6))
        for lab in np.unique(labels):
            idx = labels == lab
            ax.scatter(Xtsne[idx, 0], Xtsne[idx, 1], c=color_map.get(lab, "gray"),
                       label=lab, alpha=0.6, s=10)
        ax.legend(loc="best", fontsize=8)
        ax.set_title("t-SNE of ESM-2 650M embeddings")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "tsne_esm.png", dpi=120)
        plt.close()

        # exper vs uncond
        for lab in ["exper_real", "uncond_gen"]:
            idx = labels == lab
            if idx.any():
                plt.scatter(Xtsne[idx, 0], Xtsne[idx, 1], c=color_map.get(lab, "gray"),
                            label=lab, alpha=0.6, s=15)
        plt.legend(loc="best", fontsize=10)
        plt.title("t-SNE (ESM): exper_real vs uncond_gen")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "tsne_esm_exper_uncond.png", dpi=120)
        plt.close()

        # naive vs uncond
        for lab in ["naive_real", "uncond_gen"]:
            idx = labels == lab
            if idx.any():
                plt.scatter(Xtsne[idx, 0], Xtsne[idx, 1], c=color_map.get(lab, "gray"),
                            label=lab, alpha=0.6, s=15)
        plt.legend(loc="best", fontsize=10)
        plt.title("t-SNE (ESM): naive_real vs uncond_gen")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "tsne_esm_naive_uncond.png", dpi=120)
        plt.close()

        # PCA 2D
        pca2 = PCA(n_components=2)
        Xpca2 = pca2.fit_transform(X)
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(Xpca2[idx, 0], Xpca2[idx, 1], c=color_map.get(lab, "gray"),
                        label=lab, alpha=0.6, s=10)
        plt.legend(loc="best", fontsize=8)
        plt.title("PCA of ESM-2 650M embeddings")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pca_esm.png", dpi=120)
        plt.close()

        print(f"\n已保存: tsne_esm.png, tsne_esm_exper_uncond.png, tsne_esm_naive_uncond.png, pca_esm.png")
    except Exception as err:
        print(f"\n可视化跳过: {err}")

    print("=" * 60)
    print("ESM embedding + t-SNE 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
