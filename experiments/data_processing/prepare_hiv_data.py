#!/usr/bin/env python3
"""
HIV-1 PR 数据准备脚本
加载Stanford PR数据和训练好的Potts模型，计算统计信息
"""

import os
import numpy as np
from pathlib import Path


def load_fasta_sequences(filepath):
    """加载FASTA格式的序列文件（简单文本格式，每行一个序列）"""
    sequences = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                # 去除gap字符，保留标准氨基酸
                seq = line.replace('-', '')
                if len(seq) == 93:
                    sequences.append(seq)
    return sequences


def load_potts_model(j_path):
    """
    加载Potts模型

    J矩阵形状 (4278, 441) 的解释：
    - L = 93 (序列长度)
    - q = 21 (21种状态：20氨基酸 + gap)
    - 4278 = (93 × 92) / 2，存储上三角的位点对
    - 441 = 21 × 21
    """
    J_full = np.load(j_path)

    L = 93
    q = 21

    print(f"加载Potts模型: {j_path}")
    print(f"J矩阵形状: {J_full.shape}")

    # 重建完整的J_{ij}(a,b)张量 [L, L, q, q]
    # 假设J_full的第k行对应位点对(i,j)，k为上三角索引
    J_coupling = np.zeros((L, L, q, q))

    k = 0
    for i in range(L):
        for j in range(i+1, L):
            # J_full[k] 是(441,)，reshape成(21,21)
            J_coupling[i, j, :, :] = J_full[k].reshape(q, q)
            J_coupling[j, i, :, :] = J_full[k].reshape(q, q)  # 对称
            k += 1

    # 外场h可能需要从其他位置提取或设为零
    # 先设为零，后续根据实际数据调整
    h = np.zeros((L, q))

    print(f"重建耦合矩阵: {J_coupling.shape}")
    print(f"外场矩阵: {h.shape}")

    return {
        'J': J_coupling,
        'h': h,
        'L': L,
        'q': q
    }


def compute_pssm(sequences, num_states=21):
    """计算位置特异性评分矩阵（PSSM）"""
    L = len(sequences[0])
    pssm = np.zeros((L, num_states))

    # 氨基酸到索引的映射
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

    # 归一化
    pssm = pssm / len(sequences)

    return pssm


def compute_mutation_frequency(sequences):
    """计算每个位点的突变频率"""
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
    """计算每个位点的Shannon熵"""
    eps = 1e-10  # 避免log(0)
    entropy = -np.sum(pssm * np.log2(pssm + eps), axis=1)
    return entropy


def extract_hxb2_reference(sequences):
    """从序列中提取最接近consensus的序列作为HXB2参考"""
    # 计算consensus
    consensus_seq = []
    L = len(sequences[0])

    for pos in range(L):
        aa_counts = {}
        for seq in sequences:
            aa = seq[pos]
            aa_counts[aa] = aa_counts.get(aa, 0) + 1

        # 选择最频繁的氨基酸
        most_common_aa = max(aa_counts.items(), key=lambda x: x[1])[0]
        consensus_seq.append(most_common_aa)

    consensus = ''.join(consensus_seq)

    # 找到最接近consensus的序列
    min_dist = float('inf')
    hxb2_seq = None

    for seq in sequences:
        dist = sum(1 for a, b in zip(seq, consensus) if a != b)
        if dist < min_dist:
            min_dist = dist
            hxb2_seq = seq

    return hxb2_seq, consensus


def main():
    # 路径配置
    data_dir = Path("/mnt/hbnas/home/pfp/hiv/discrete_flow_models/data/hiv_msa")
    potts_dir = Path("/mnt/hbnas/home/pfp/hiv/Mi3-GPU/examples")
    output_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/processed")
    reference_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/reference")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: HIV-1 PR 数据准备")
    print("=" * 60)

    # 1. 加载val set数据（仅val用于评估）
    print("\n[1/5] 加载val set数据...")

    naive_val_seqs = load_fasta_sequences(data_dir / "pr_naive/val_sequences.txt")
    exper_val_seqs = load_fasta_sequences(data_dir / "pr_exper/val_sequences.txt")

    print(f"  Naive val: {len(naive_val_seqs)} 条序列")
    print(f"  Exper val: {len(exper_val_seqs)} 条序列")

    # 验证序列长度
    if naive_val_seqs:
        print(f"  Naive序列长度: {len(naive_val_seqs[0])} aa")
    if exper_val_seqs:
        print(f"  Exper序列长度: {len(exper_val_seqs[0])} aa")

    # 2. 加载Potts模型
    print("\n[2/5] 加载Potts模型...")

    potts_naive_path = potts_dir / "pr.naive.splitseq/pr_naive_inference/run_63/J.npy"
    potts_exper_path = potts_dir / "pr.exper.splitseq/pr_exper_inference/run_63/J.npy"

    potts_naive = load_potts_model(potts_naive_path)
    potts_exper = load_potts_model(potts_exper_path)

    # 3. 计算统计信息（基于val set）
    print("\n[3/5] 计算统计信息...")

    # Naive统计
    print("  计算Naive统计...")
    naive_pssm = compute_pssm(naive_val_seqs)
    naive_mutation_freq = compute_mutation_frequency(naive_val_seqs)
    naive_entropy = compute_shannon_entropy(naive_pssm)

    print(f"    平均Shannon熵: {naive_entropy.mean():.3f}")
    print(f"    最高熵位点: {np.argmax(naive_entropy)} (值: {naive_entropy.max():.3f})")
    print(f"    最低熵位点: {np.argmin(naive_entropy)} (值: {naive_entropy.min():.3f})")

    # Exper统计
    print("  计算Exper统计...")
    exper_pssm = compute_pssm(exper_val_seqs)
    exper_mutation_freq = compute_mutation_frequency(exper_val_seqs)
    exper_entropy = compute_shannon_entropy(exper_pssm)

    print(f"    平均Shannon熵: {exper_entropy.mean():.3f}")
    print(f"    最高熵位点: {np.argmax(exper_entropy)} (值: {exper_entropy.max():.3f})")
    print(f"    最低熵位点: {np.argmin(exper_entropy)} (值: {exper_entropy.min():.3f})")

    # 4. 提取HXB2参考序列
    print("\n[4/5] 提取HXB2参考序列...")

    # 使用naive val set提取HXB2（naive更保守）
    hxb2_naive, naive_consensus = extract_hxb2_reference(naive_val_seqs)
    hxb2_exper, exper_consensus = extract_hxb2_reference(exper_val_seqs)

    print(f"  Naive consensus: {naive_consensus}")
    print(f"  Naive HXB2代理: {hxb2_naive}")
    print(f"  Exper consensus: {exper_consensus}")
    print(f"  Exper HXB2代理: {hxb2_exper}")

    # 使用naive的HXB2作为主参考
    hxb2_ref = hxb2_naive

    # 5. 保存所有数据
    print("\n[5/5] 保存数据...")

    # 保存val set序列
    def save_fasta(seqs, filepath):
        with open(filepath, 'w') as f:
            for i, seq in enumerate(seqs):
                f.write(f">SEQ_{i}\n{seq}\n")

    save_fasta(naive_val_seqs, output_dir / "pr_naive_val.fasta")
    save_fasta(exper_val_seqs, output_dir / "pr_exper_val.fasta")

    # 保存HXB2参考
    with open(reference_dir / "hxb2_pr.fasta", 'w') as f:
        f.write(f">HXB2_PR_Reference\n{hxb2_ref}\n")

    # 保存Potts模型
    np.savez_compressed(
        output_dir / "potts_models.npz",
        potts_naive_J=potts_naive['J'],
        potts_naive_h=potts_naive['h'],
        potts_exper_J=potts_exper['J'],
        potts_exper_h=potts_exper['h'],
        potts_L=potts_naive['L'],
        potts_q=potts_naive['q']
    )

    # 保存统计信息
    np.savez_compressed(
        output_dir / "pr_statistics.npz",
        naive_pssm=naive_pssm,
        exper_pssm=exper_pssm,
        naive_entropy=naive_entropy,
        exper_entropy=exper_entropy,
        hxb2_ref=np.array(list(hxb2_ref), dtype='U1')
    )

    print(f"\n保存完成！")
    print(f"  Val sequences: {output_dir / 'pr_naive_val.fasta'}, {output_dir / 'pr_exper_val.fasta'}")
    print(f"  HXB2 reference: {reference_dir / 'hxb2_pr.fasta'}")
    print(f"  Potts models: {output_dir / 'potts_models.npz'}")
    print(f"  Statistics: {output_dir / 'pr_statistics.npz'}")

    print("\n" + "=" * 60)
    print("Phase 1 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
