#!/usr/bin/env python3
"""
AAR（Amino Acid Retention）计算脚本
计算生成序列对HXB2的保留率。
与 Phase2 一致：生成序列为 99 aa，93aa = 99aa[5:98]；HXB2 为 93 aa。
"""

import numpy as np
from pathlib import Path

# Phase2 约定：DPLM 生成 99 aa，93aa = 99aa[5:98]
DPLM_LEN = 99
PR_93_START, PR_93_END = 5, 98


def to_93aa_if_99(seq):
    """若为 99 aa 生成序列，取 93aa 部分与 HXB2(93aa) 对齐。"""
    if len(seq) == DPLM_LEN:
        return seq[PR_93_START:PR_93_END]
    return seq


def load_sequences(fasta_path):
    """加载FASTA序列"""
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
    计算AAR（Amino Acid Retention）

    Args:
        seq: 待计算序列（93aa）
        hxb2_seq: HXB2参考序列（93aa）

    Returns:
        aar: 保留率 [0, 1]
    """
    if len(seq) != len(hxb2_seq):
        print(f"警告：序列长度不匹配 {len(seq)} vs {len(hxb2_seq)}")
        return np.nan

    matches = sum(1 for a, b in zip(seq, hxb2_seq) if a == b)
    aar = matches / len(seq)

    return aar


def main():
    # 路径配置
    data_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/processed")
    generated_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/generated")
    output_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/aar")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Phase 4.1: AAR计算")
    print("="*60)

    # 1. 加载HXB2参考序列
    print("\n[1/3] 加载HXB2参考序列...")

    with open(data_dir / "../reference/hxb2_pr.fasta", 'r') as f:
        for line in f:
            if not line.startswith('>'):
                hxb2_seq = line.strip()
                break

    print(f"HXB2长度: {len(hxb2_seq)} aa")
    print(f"HXB2: {hxb2_seq}")

    # 2. 计算真实序列的AAR
    print("\n[2/3] 计算真实序列AAR...")

    _, naive_seqs = load_sequences(data_dir / "pr_naive_val.fasta")
    _, exper_seqs = load_sequences(data_dir / "pr_exper_val.fasta")

    print(f"  Naive: {len(naive_seqs)} 条")
    print(f"  Exper: {len(exper_seqs)} 条")

    naive_aars = [calc_aar(seq, hxb2_seq) for seq in naive_seqs]
    exper_aars = [calc_aar(seq, hxb2_seq) for seq in exper_seqs]

    print(f"  Naive AAR均值: {np.nanmean(naive_aars)*100:.2f}%")
    print(f"  Exper AAR均值: {np.nanmean(exper_aars)*100:.2f}%")

    # 3. 计算生成序列的AAR（Phase2 为 99 aa，取 93aa 与 HXB2 比较）
    print("\n[3/3] 计算生成序列AAR...")

    _, gen_uncond_seqs = load_sequences(generated_dir / "pr_uncond.fasta")
    _, gen_naive_potts_seqs = load_sequences(generated_dir / "pr_naive_potts.fasta")
    _, gen_exper_potts_seqs = load_sequences(generated_dir / "pr_exper_potts.fasta")

    gen_uncond_seqs = [to_93aa_if_99(s) for s in gen_uncond_seqs]
    gen_naive_potts_seqs = [to_93aa_if_99(s) for s in gen_naive_potts_seqs]
    gen_exper_potts_seqs = [to_93aa_if_99(s) for s in gen_exper_potts_seqs]

    print(f"  Uncond: {len(gen_uncond_seqs)} 条 (93aa)")
    print(f"  Naive-Potts: {len(gen_naive_potts_seqs)} 条 (93aa)")
    print(f"  Exper-Potts: {len(gen_exper_potts_seqs)} 条 (93aa)")

    gen_uncond_aars = [calc_aar(seq, hxb2_seq) for seq in gen_uncond_seqs]
    gen_naive_potts_aars = [calc_aar(seq, hxb2_seq) for seq in gen_naive_potts_seqs]
    gen_exper_potts_aars = [calc_aar(seq, hxb2_seq) for seq in gen_exper_potts_seqs]

    print(f"  Uncond AAR均值: {np.nanmean(gen_uncond_aars)*100:.2f}%")
    print(f"  Naive-Potts AAR均值: {np.nanmean(gen_naive_potts_aars)*100:.2f}%")
    print(f"  Exper-Potts AAR均值: {np.nanmean(gen_exper_potts_aars)*100:.2f}%")

    # 4. 保存结果
    print("\n保存结果...")

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

    # 统计汇总
    print("\n统计汇总：")

    summary = {
        'naive_real_mean': np.nanmean(naive_aars),
        'exper_real_mean': np.nanmean(exper_aars),
        'uncond_gen_mean': np.nanmean(gen_uncond_aars),
        'naive_potts_gen_mean': np.nanmean(gen_naive_potts_aars),
        'exper_potts_gen_mean': np.nanmean(gen_exper_potts_aars),
    }

    for key, value in summary.items():
        print(f"  {key}: {value*100:.2f}%")

    # 检查是否在40-60%区间
    print("\nAAR验证（40-60%区间）：")

    for key in ['uncond_gen_mean', 'naive_potts_gen_mean', 'exper_potts_gen_mean']:
        value = summary[key]
        in_range = 0.4 <= value <= 0.6
        status = "✓ 通过" if in_range else "✗ 未通过"
        print(f"  {key}: {value*100:.2f}% - {status}")

    with open(output_dir / "aar_summary.txt", 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value:.4f}\n")

    # 可视化：AAR 分布（按 type 分组箱线图）
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
        print(f"  图表: {output_dir / 'aar_distribution.png'}")
    except Exception as e:
        print("  AAR 可视化跳过:", e)

    print(f"\n保存完成！")
    print(f"  详细结果: {output_dir / 'aar_all.csv'}")
    print(f"  统计汇总: {output_dir / 'aar_summary.txt'}")

    print("\n" + "="*60)
    print("Phase 4.1 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
