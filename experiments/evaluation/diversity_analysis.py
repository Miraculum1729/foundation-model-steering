#!/usr/bin/env python3
"""
Phase 4.2: 多样性分析
指标：序列间平均 Hamming 距离、每位点 Shannon 熵、唯一序列比例。
与 Phase2 一致：生成序列为 99 aa，93aa = 99aa[5:98]。
"""

import numpy as np
from pathlib import Path
from collections import Counter

# Phase2 约定
DPLM_LEN = 99
PR_93_START, PR_93_END = 5, 98

# 大集合做 Hamming 时随机采样对数，避免 O(N^2) 过慢
MAX_PAIRS_FOR_HAMMING = 5000


def to_93aa_if_99(seq):
    """若为 99 aa 生成序列，取 93aa 部分。"""
    if len(seq) == DPLM_LEN:
        return seq[PR_93_START:PR_93_END]
    return seq


def load_sequences(fasta_path):
    """加载 FASTA 序列。"""
    sequences = []
    names = []
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


def mean_hamming(sequences, max_pairs=MAX_PAIRS_FOR_HAMMING):
    """序列间平均 Hamming 距离（可采样以控制时间）。"""
    n = len(sequences)
    if n < 2:
        return np.nan
    L = len(sequences[0])
    if max_pairs is None or n * (n - 1) // 2 <= max_pairs:
        # 全量
        total, count = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += sum(a != b for a, b in zip(sequences[i], sequences[j]))
                count += 1
        return total / count / L if count else np.nan
    # 随机采样 max_pairs 对
    rng = np.random.default_rng(42)
    total, count = 0, 0
    for _ in range(max_pairs):
        i, j = rng.integers(0, n), rng.integers(0, n)
        if i >= j:
            j = (j + 1) % n
        if i == j:
            continue
        total += sum(a != b for a, b in zip(sequences[i], sequences[j]))
        count += 1
    return total / count / L if count else np.nan


def shannon_entropy_per_position(sequences):
    """每位点 Shannon 熵 (nat)。"""
    if not sequences:
        return None
    L = len(sequences[0])
    entropies = []
    for pos in range(L):
        counts = Counter(s[pos] for s in sequences if pos < len(s))
        n = sum(counts.values())
        if n == 0:
            entropies.append(0.0)
            continue
        h = 0.0
        for c in counts.values():
            p = c / n
            if p > 0:
                h -= p * np.log(p)
        entropies.append(h)
    return np.array(entropies)


def unique_ratio(sequences):
    """唯一序列比例。"""
    if not sequences:
        return np.nan
    return len(set(sequences)) / len(sequences)


def main():
    data_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/processed")
    generated_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/generated")
    output_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/diversity")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 4.2: 多样性分析")
    print("=" * 60)

    # 真实序列（已是 93 aa）
    _, naive_seqs = load_sequences(data_dir / "pr_naive_val.fasta")
    _, exper_seqs = load_sequences(data_dir / "pr_exper_val.fasta")

    # 生成序列（99 aa → 93 aa）
    _, gen_uncond = load_sequences(generated_dir / "pr_uncond.fasta")
    _, gen_naive_potts = load_sequences(generated_dir / "pr_naive_potts.fasta")
    _, gen_exper_potts = load_sequences(generated_dir / "pr_exper_potts.fasta")
    gen_uncond = [to_93aa_if_99(s) for s in gen_uncond]
    gen_naive_potts = [to_93aa_if_99(s) for s in gen_naive_potts]
    gen_exper_potts = [to_93aa_if_99(s) for s in gen_exper_potts]

    # Potts MCMC 纯采样序列（若存在）
    potts_mcmc_naive = []
    potts_mcmc_exper = []
    if (generated_dir / "pr_potts_mcmc_naive.fasta").exists():
        _, s = load_sequences(generated_dir / "pr_potts_mcmc_naive.fasta")
        potts_mcmc_naive = [to_93aa_if_99(x) for x in s]
    if (generated_dir / "pr_potts_mcmc_exper.fasta").exists():
        _, s = load_sequences(generated_dir / "pr_potts_mcmc_exper.fasta")
        potts_mcmc_exper = [to_93aa_if_99(x) for x in s]

    sets = [
        ("naive_real", naive_seqs),
        ("exper_real", exper_seqs),
        ("uncond_gen", gen_uncond),
        ("naive_potts_gen", gen_naive_potts),
        ("exper_potts_gen", gen_exper_potts),
        ("potts_mcmc_naive", potts_mcmc_naive),
        ("potts_mcmc_exper", potts_mcmc_exper),
    ]

    results = []
    for name, seqs in sets:
        if not seqs:
            results.append({"group": name, "n": 0, "mean_hamming": np.nan, "mean_entropy": np.nan, "unique_ratio": np.nan})
            continue
        mean_ham = mean_hamming(seqs)
        ent = shannon_entropy_per_position(seqs)
        mean_ent = float(np.nanmean(ent)) if ent is not None else np.nan
        ur = unique_ratio(seqs)
        results.append({
            "group": name,
            "n": len(seqs),
            "mean_hamming": mean_ham,
            "mean_entropy": mean_ent,
            "unique_ratio": ur,
        })
        print(f"  {name}: n={len(seqs)}, mean_hamming={mean_ham:.4f}, mean_entropy={mean_ent:.4f}, unique_ratio={ur:.4f}")

    # 保存汇总
    with open(output_dir / "diversity_summary.txt", "w") as f:
        f.write("group\tn\tmean_hamming\tmean_entropy\tunique_ratio\n")
        for r in results:
            f.write(f"{r['group']}\t{r['n']}\t{r['mean_hamming']:.6f}\t{r['mean_entropy']:.6f}\t{r['unique_ratio']:.6f}\n")

    # 保存每位点熵（以第一个集合的长度为准，这里都是 93）
    for name, seqs in sets:
        if not seqs:
            continue
        ent = shannon_entropy_per_position(seqs)
        if ent is not None:
            np.savetxt(output_dir / f"entropy_per_position_{name}.txt", ent, fmt="%.6f")

    # 可视化：hamming_distances.png, entropy_by_site.png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = [r["group"] for r in results if r["n"] > 0]
        hams = [r["mean_hamming"] for r in results if r["n"] > 0]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(groups, hams, color="steelblue", edgecolor="black")
        ax.set_ylabel("Mean pairwise Hamming distance (norm.)")
        ax.set_title("Diversity: mean Hamming distance by group")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "hamming_distances.png", dpi=120)
        plt.close()

        # entropy_by_site: 柱状图，每组一子图，自上而下排列，每组最高 10 个位点深色其余浅色
        L = 93
        plot_sets = [(name, seqs) for name, seqs in sets if seqs]
        n_plots = len(plot_sets)
        if n_plots > 0:
            # 先计算全体熵以统一 y 轴范围
            all_ent = []
            for name, seqs in plot_sets:
                ent = shannon_entropy_per_position(seqs)
                if ent is not None:
                    all_ent.extend(ent)
            y_max = max(all_ent) * 1.05 if all_ent else 1.0
            y_min = 0.0

            fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=False, sharey=False)
            if n_plots == 1:
                axes = [axes]
            colors = {"naive_real": "#1f77b4", "exper_real": "#ff7f0e", "uncond_gen": "#2ca02c",
                      "naive_potts_gen": "#d62728", "exper_potts_gen": "#9467bd",
                      "potts_mcmc_naive": "#17becf", "potts_mcmc_exper": "#e377c2"}
            for ax, (name, seqs) in zip(axes, plot_sets):
                ent = shannon_entropy_per_position(seqs)
                if ent is None:
                    continue
                ax.set_ylim(y_min, y_max)
                base_color = colors.get(name, "#7f7f7f")
                import matplotlib.colors as mcolors
                rgb = mcolors.to_rgba(base_color)
                light_color = (*rgb[:3], 0.3)
                dark_color = base_color
                # 最高 10 个位点用深色，其余浅色
                top10_idx = np.argsort(ent)[-10:]
                top10_pos_1based = sorted(top10_idx + 1)  # 1-based 位点，排序便于横轴
                bar_colors = [dark_color if i in top10_idx else light_color for i in range(L)]
                x = np.arange(1, L + 1)
                ax.bar(x, ent, color=bar_colors, edgecolor="none")
                ax.set_ylabel("Shannon entropy")
                ax.set_title(name)
                ax.set_xlim(0.5, L + 0.5)
                # 仅在深色柱子下标出 site 数字，旋转避免重合
                ax.set_xticks(top10_pos_1based)
                ax.set_xticklabels([str(p) for p in top10_pos_1based], fontsize=8, rotation=45, ha="right")
            axes[-1].set_xlabel("Position (1-based PR, top 10 sites labeled)")
            fig.suptitle("Entropy by site (top 10 sites per group in dark color)", fontsize=10, y=1.02)
            plt.tight_layout()
            plt.savefig(output_dir / "entropy_by_site.png", dpi=120, bbox_inches="tight")
            plt.close()
        print("  图表: hamming_distances.png, entropy_by_site.png")
    except Exception as e:
        print("  可视化跳过:", e)

    print(f"\n保存完成: {output_dir}")
    print("  diversity_summary.txt, entropy_per_position_*.txt")
    print("=" * 60)
    print("Phase 4.2 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
