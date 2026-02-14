#!/usr/bin/env python3
"""
HIV-1 PR DPLM Potts引导生成脚本
实现HXB2模板clamping + Potts能量引导
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from pprint import pprint

# 添加DPLM模块路径
sys.path.insert(0, '/mnt/hbnas/home/pfp/hiv2026/dplm/src')

os.environ["BYPROT_SKIP_DATAMODULES"] = "1"
os.environ["BYPROT_SKIP_TASKS"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/hbnas/home/pfp/.cache/huggingface")

from byprot.models.dplm.dplm import DiffusionProteinLanguageModel


def compute_potts_energy(seq, potts_model):
    """
    计算序列的Potts能量

    Args:
        seq: 氨基酸序列（93 aa）
        potts_model: {'J': ..., 'h': ..., 'aa_to_idx': ...}

    Returns:
        energy: 标量能量值
    """
    J = potts_model['J']  # [L, L, q, q]
    h = potts_model['h']  # [L, q]
    L = potts_model['L']
    q = potts_model['q']

    # 氨基酸到索引的映射
    aa_to_idx = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
    }

    energy = 0.0

    # 外场项
    for i in range(L):
        aa = seq[i] if i < len(seq) else 'A'  # 处理长度不足
        if aa in aa_to_idx:
            idx_i = i * q + aa_to_idx[aa]
            energy -= h[i, aa_to_idx[aa]]

    # 耦合项
    for i in range(L):
        aa_i = seq[i] if i < len(seq) else 'A'
        idx_i = i * q + aa_to_idx[aa_i] if aa_i in aa_to_idx else i * q
        for j in range(i+1, L):
            aa_j = seq[j] if j < len(seq) else 'A'
            idx_j = j * q + aa_to_idx[aa_j] if aa_j in aa_to_idx else j * q
            if aa_i in aa_to_idx and aa_j in aa_to_idx:
                energy -= J[i, j, aa_to_idx[aa_i], aa_to_idx[aa_j]]

    return energy


def potts_guided_sampling_step(
    model,
    current_tokens,
    fixed_mask,
    potts_model,
    beta=1.0,
    tokenizer=None
):
    """
    在DPLM的每一步反向扩散中，使用Potts能量修正采样分布

    Args:
        model: DPLM模型
        current_tokens: 当前token序列 [batch, seq_len]
        fixed_mask: 固定位置mask [batch, seq_len], True=固定
        potts_model: Potts模型 (J, h)
        beta: 引导强度参数
        tokenizer: 用于映射token到氨基酸

    Returns:
        new_tokens: 更新后的token序列
    """
    with torch.no_grad():
        logits = model(current_tokens)

    batch_size, seq_len, vocab_size = logits.shape

    # 与 DPLM forward_decoder 一致：屏蔽特殊 token，否则会采到 pad/bos/eos/mask 等，解码成 <null_1> 等非法字符
    logits = logits.clone()
    logits[..., model.mask_id] = -1e9
    logits[..., model.x_id] = -1e9
    logits[..., model.pad_id] = -1e9
    logits[..., model.bos_id] = -1e9
    logits[..., model.eos_id] = -1e9

    dplm_probs = torch.softmax(logits, dim=-1)
    # TODO: Potts 能量修正 dplm_probs（当前未实现，仅用 DPLM 分布）

    new_tokens = torch.multinomial(dplm_probs.view(-1, vocab_size), 1)
    new_tokens = new_tokens.view(batch_size, seq_len)

    # 恢复固定位置
    new_tokens = torch.where(fixed_mask, current_tokens, new_tokens)

    return new_tokens


# DPLM 生成 99 aa。用 99 aa HXB2 模板固定 motif：PQITL(0-4), DTG(24-26), GGIG(47-50), CTLNF(94-98)；93aa=99aa[5:98]
DPLM_PR_START = 5   # 99aa 中 93aa PR 起始下标
DPLM_LEN = 99       # DPLM 序列长度

# Potts 使用 ALPHA21：0=gap, 1-20=ACDEFGHIKLMNPQRSTVWY（与 Mi3 常见一致）
ALPHA21 = "-ACDEFGHIKLMNPQRSTVWY"


def _get_aa_to_token_id(tokenizer, device):
    """构建 20 个氨基酸字符到 DPLM token id 的映射。"""
    aa_to_token_id = {}
    for c in ALPHA21[1:]:  # 跳过 gap
        enc = tokenizer.encode(c, add_special_tokens=False)
        if enc:
            aa_to_token_id[c] = enc[0]
    return aa_to_token_id


def _build_token_to_potts(tokenizer, device, max_id=256):
    """一次构建：token_id -> potts_idx (0..20)，非氨基酸为 -1。"""
    arr = torch.full((max_id,), -1, dtype=torch.long, device=device)
    for i, c in enumerate(ALPHA21):
        if c == "-":
            continue
        enc = tokenizer.encode(c, add_special_tokens=False)
        if enc and enc[0] < max_id:
            arr[enc[0]] = i
    return arr


def compute_potts_Ei_bias(
    output_tokens,
    output_masks,
    J_gpu,
    h_gpu,
    token_to_potts,
    aa_to_token_id,
    vocab_size,
    L,
    q,
    beta,
    device,
):
    """
    向量化 + GPU：F_i(a)=h_i(a)+sum_{j已解码,j!=i} J_ij(a,x_j)，bias = -beta*F。
    J/h 常驻 GPU；单次调用用 gather + sum 无 Python 内层循环。
    """
    B = output_tokens.shape[0]
    L99 = output_tokens.shape[1]

    tokens_93 = output_tokens[:, 5:5 + L]
    decoded = token_to_potts[tokens_93.clamp(0, token_to_potts.shape[0] - 1)].clone()
    decoded[(tokens_93 < 0) | (tokens_93 >= token_to_potts.shape[0])] = -1
    valid = (decoded >= 0).float()
    idx = decoded.clamp(0)

    # J_gathered[i,j,a,b] = J[i,j,a, decoded[b,j]]，index (L,L,q,B)
    index = idx.t().unsqueeze(0).unsqueeze(2).expand(L, L, q, -1).long()
    J_gathered = torch.gather(J_gpu, 3, index)
    valid_j = valid.t().unsqueeze(0).unsqueeze(2).expand(L, L, q, -1)
    off_diag = (1 - torch.eye(L, device=device, dtype=J_gathered.dtype)).unsqueeze(2).unsqueeze(3).expand(L, L, q, B)
    F = h_gpu.unsqueeze(0).expand(B, -1, -1) + (
        (J_gathered * valid_j * off_diag).sum(dim=1).permute(2, 0, 1)
    )

    bias = torch.zeros((B, L99, vocab_size), device=device, dtype=torch.float32)
    om = output_masks[:, 5:5 + L]
    for aa_char, token_id in aa_to_token_id.items():
        if token_id >= vocab_size:
            continue
        potts_a = ALPHA21.index(aa_char)
        val = -beta * F[:, :, potts_a]
        bias[:, 5:5 + L, token_id] = torch.where(om, val, bias[:, 5:5 + L, token_id])

    return bias.to(output_tokens.dtype)


def make_potts_logit_modifier(potts_model, beta, tokenizer, model, device, bias_clip=None):
    """返回可在 forward_decoder 中调用的 logit_modifier。J/h 一次放入 GPU，计算向量化。

    bias_clip: 若给定正数，将 Potts 对 logits 的调整裁剪到 [-bias_clip, bias_clip]，
               避免过度拉向离群点、使生成更贴近真实主体分布。
    """
    L = int(potts_model["L"])
    q = int(potts_model["q"])
    J_gpu = torch.from_numpy(potts_model["J"]).to(device=device, dtype=torch.float32)
    h_gpu = torch.from_numpy(potts_model["h"]).to(device=device, dtype=torch.float32)
    token_to_potts = _build_token_to_potts(tokenizer, device)
    aa_to_token_id = _get_aa_to_token_id(tokenizer, device)

    _chunk = 128  # 大 batch 分块，避免 (L,L,q,B) 显存与算力爆炸

    def logit_modifier(logits, output_tokens, output_masks):
        B = logits.shape[0]
        V = logits.shape[-1]
        if B <= _chunk:
            potts_bias = compute_potts_Ei_bias(
                output_tokens, output_masks,
                J_gpu, h_gpu, token_to_potts, aa_to_token_id,
                vocab_size=V, L=L, q=q, beta=beta, device=device,
            )
        else:
            parts = []
            for start in range(0, B, _chunk):
                end = min(start + _chunk, B)
                part_bias = compute_potts_Ei_bias(
                    output_tokens[start:end], output_masks[start:end],
                    J_gpu, h_gpu, token_to_potts, aa_to_token_id,
                    vocab_size=V, L=L, q=q, beta=beta, device=device,
                )
                parts.append(part_bias)
            potts_bias = torch.cat(parts, dim=0)
        if bias_clip is not None and bias_clip > 0:
            potts_bias = potts_bias.clamp(-bias_clip, bias_clip)
        return logits + potts_bias

    return logit_modifier


def build_hxb2_99_from_93(hxb2_93):
    """
    从 93 aa HXB2 PR 构建 99 aa 模板：PQITL(0-4) + 93aa(5-97) + F(98)；93aa=99aa[5:98]。
    """
    if len(hxb2_93) != 93:
        raise ValueError(f"HXB2 93aa 长度应为 93，得到 {len(hxb2_93)}")
    # 99 aa = 5 (PQITL) + 93 aa + 1 (F)
    hxb2_99 = "PQITL" + hxb2_93 + "F"
    assert len(hxb2_99) == 99
    return hxb2_99


def _apply_motif_99aa(seq_99_list, hxb2_99_list):
    """
    用 99 aa HXB2 模板在 99 aa 序列上固定 motif。
    hxb2_99_list 必须是长度 99 的列表。
    固定：PQITL(0-4), DTG(24-26), GGIG(47-50), CTLNF(94-98)
    """
    if len(hxb2_99_list) != 99:
        raise ValueError(f"HXB2 99aa 模板长度应为 99，得到 {len(hxb2_99_list)}")
    seq_99_list[0:5] = hxb2_99_list[0:5]      # PQITL
    seq_99_list[24:27] = hxb2_99_list[24:27]  # DTG
    seq_99_list[47:51] = hxb2_99_list[47:51]  # GGIG
    seq_99_list[94:99] = hxb2_99_list[94:99]  # CTLNF


def generate_unconditioned(model, tokenizer, hxb2_template_99, num_samples=500, max_iter=500, device='cuda'):
    """无条件生成：用 99 aa HXB2 模板固定 motif (PQITL, DTG, GGIG, CTLNF)。"""
    print("\n" + "="*60)
    print("生成Unconditioned序列...")
    print("="*60)

    seq_list = ['<mask>'] * DPLM_LEN
    hxb2_list = list(hxb2_template_99)
    _apply_motif_99aa(seq_list, hxb2_list)

    seq = [''.join(seq_list)]
    init_seq = seq * num_samples

    batch = tokenizer.batch_encode_plus(
        init_seq,
        add_special_tokens=True,
        padding='longest',
        return_tensors='pt',
    )

    batch = {
        "input_ids": batch["input_ids"],
        "input_mask": batch["attention_mask"].bool(),
    }

    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"输入序列示例: {seq[0][:50]}...")
    print(f"序列长度: 99 aa; 固定: PQITL(0-4), DTG(24-26), GGIG(47-50), CTLNF(94-98); 93aa=99aa[5:98]")

    input_tokens = batch["input_ids"]

    # 创建partial_mask：非mask位置为True（固定），与 generate_dplm.py 一致
    partial_mask = input_tokens.ne(model.mask_id)

    # 使用标准DPLM生成
    print("开始生成...")
    with torch.amp.autocast('cuda'):
        outputs = model.generate(
            input_tokens=input_tokens,
            tokenizer=tokenizer,
            max_iter=max_iter,
            sampling_strategy='gumbel_argmax',
            partial_masks=partial_mask,
        )

    output_tokens = outputs

    print("生成完成！")

    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )
    ]

    return output_results


def generate_potts_guided(model, tokenizer, hxb2_template_99, potts_model,
                       beta=1.0, num_samples=500, max_iter=500,
                       device='cuda', output_name="potts", bias_clip=None):
    """Potts引导生成。用 99 aa HXB2 模板固定 motif。

    beta: 引导强度，越小越贴近 DPLM/真实主体分布（建议 0.2–0.5 贴近主体）。
    bias_clip: 对 logits 偏置做裁剪上限，避免过度拉向离群点（如 2.0）。
    """
    print("\n" + "="*60)
    print(f"生成{output_name.upper()}引导序列 (beta={beta}" + (f", bias_clip={bias_clip})" if bias_clip else ")") + "...")
    print("="*60)

    seq_list = ['<mask>'] * DPLM_LEN
    hxb2_list = list(hxb2_template_99)
    _apply_motif_99aa(seq_list, hxb2_list)

    seq = [''.join(seq_list)]
    init_seq = seq * num_samples

    batch = tokenizer.batch_encode_plus(
        init_seq,
        add_special_tokens=True,
        padding='longest',
        return_tensors='pt',
    )

    batch = {
        "input_ids": batch["input_ids"],
        "input_mask": batch["attention_mask"].bool(),
    }

    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"输入序列示例: {seq[0][:50]}...")
    print(f"序列长度: 99 aa; 固定: PQITL(0-4), DTG(24-26), GGIG(47-50), CTLNF(94-98); beta={beta}" + (f", bias_clip={bias_clip}" if bias_clip else ""))

    input_tokens = batch["input_ids"]
    partial_mask = input_tokens.ne(model.mask_id)

    logit_modifier = make_potts_logit_modifier(potts_model, beta, tokenizer, model, device, bias_clip=bias_clip)
    print("开始 Potts 引导生成（E_i(aa) 偏置 logits）...")
    with torch.amp.autocast('cuda'):
        output_tokens = model.generate(
            input_tokens=input_tokens,
            tokenizer=tokenizer,
            max_iter=max_iter,
            sampling_strategy='gumbel_argmax',
            partial_masks=partial_mask,
            logit_modifier=logit_modifier,
        )

    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )
    ]

    print("生成完成！")
    return output_results


def save_fasta(sequences, saveto_name, prefix="SEQUENCE"):
    """保存序列到FASTA文件"""
    os.makedirs(os.path.dirname(saveto_name), exist_ok=True)
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(sequences):
        fp_save.write(f">{prefix}_{idx}\n{seq}\n")
    fp_save.close()
    print(f"保存到: {saveto_name}")


def main(num_samples=500, max_iter=500, beta_naive=0.3, beta_exper=0.5, bias_clip=2.0, uncond_only=False, output_suffix=None, potts_only=False):
    # 参数设置
    model_name = "airkingbd/dplm_150m"

    # 路径配置（支持从仓库根目录或 experiments/generation 运行）
    repo_root = Path(__file__).resolve().parents[2]
    potts_models_path = repo_root / "hiv_data/processed/potts_models.npz"
    hxb2_path = repo_root / "hiv_data/reference/hxb2_pr.fasta"
    output_dir = repo_root / "exp_results/generated"

    if not hxb2_path.exists():
        raise FileNotFoundError(
            f"HXB2 参考序列不存在: {hxb2_path}\n请先运行 Phase 1: python experiments/data_processing/prepare_hiv_data.py"
        )
    if not potts_models_path.exists():
        raise FileNotFoundError(
            f"Potts 模型不存在: {potts_models_path}\n请先运行 Phase 1: python experiments/data_processing/prepare_hiv_data.py"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    sfx = f"_{output_suffix}" if output_suffix else ""
    def out(name):
        return output_dir / f"{name}{sfx}.fasta"

    # 加载 93 aa HXB2，构建 99 aa 模板（PQITL + 93aa + F）
    print("加载HXB2参考序列...")
    with open(hxb2_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                hxb2_93 = line.strip()
                break

    if len(hxb2_93) != 93:
        raise ValueError(f"HXB2 参考应为 93 aa，得到 {len(hxb2_93)}")
    hxb2_template_99 = build_hxb2_99_from_93(hxb2_93)
    print(f"HXB2 93 aa → 99 aa 模板")
    print(f"  99aa 模板: {hxb2_template_99[:20]}...{hxb2_template_99[-10:]}")

    # 加载Potts模型
    print("\n加载Potts模型...")
    potts_data = np.load(potts_models_path)

    potts_naive = {
        'J': potts_data['potts_naive_J'],
        'h': potts_data['potts_naive_h'],
        'L': int(potts_data['potts_L']),
        'q': int(potts_data['potts_q'])
    }

    potts_exper = {
        'J': potts_data['potts_exper_J'],
        'h': potts_data['potts_exper_h'],
        'L': int(potts_data['potts_L']),
        'q': int(potts_data['potts_q'])
    }

    # 加载DPLM模型
    print("\n加载DPLM模型...")
    model = DiffusionProteinLanguageModel.from_pretrained(model_name)
    tokenizer = model.tokenizer
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"DPLM模型: {model_name}")
    print(f"设备: {device}")

    print("\n" + "="*60)
    print("Phase 2: DPLM Potts引导生成")
    print("="*60)

    uncond_sequences = []
    if not potts_only:
        # 1. Unconditioned生成
        print("\n[1/3] Unconditioned生成...")
        uncond_sequences = generate_unconditioned(
            model, tokenizer, hxb2_template_99,
            num_samples=num_samples, max_iter=max_iter, device=device
        )
        save_fasta(uncond_sequences, out("pr_uncond"), prefix="UNCOND")

    if uncond_only:
        print("\n" + "="*60)
        print("Phase 2 完成（仅 Uncond）！")
        print("="*60)
        print(f"\nUncond: {len(uncond_sequences)} 条序列 -> {out('pr_uncond')}")
        return

    # 2. Naive-Potts引导生成
    step2, step3 = ("[1/2]", "[2/2]") if potts_only else ("[2/3]", "[3/3]")
    print(f"\n{step2} Naive-Potts引导生成...")
    naive_potts_sequences = generate_potts_guided(
        model, tokenizer, hxb2_template_99, potts_naive,
        beta=beta_naive, num_samples=num_samples, max_iter=max_iter,
        device=device, output_name="naive_potts", bias_clip=bias_clip
    )
    save_fasta(naive_potts_sequences, out("pr_naive_potts"), prefix="NAIVE_POTTS")

    # 3. Exper-Potts引导生成
    print(f"\n{step3} Exper-Potts引导生成...")
    exper_potts_sequences = generate_potts_guided(
        model, tokenizer, hxb2_template_99, potts_exper,
        beta=beta_exper, num_samples=num_samples, max_iter=max_iter,
        device=device, output_name="exper_potts", bias_clip=bias_clip
    )
    save_fasta(exper_potts_sequences, out("pr_exper_potts"), prefix="EXPER_POTTS")

    print("\n" + "="*60)
    print("Phase 2 完成！")
    print("="*60)
    print("\n生成统计：")
    print(f"  Uncond: {len(uncond_sequences)} 条序列")
    print(f"  Naive-Potts: {len(naive_potts_sequences)} 条序列")
    print(f"  Exper-Potts: {len(exper_potts_sequences)} 条序列")
    print(f"  总计: {len(uncond_sequences) + len(naive_potts_sequences) + len(exper_potts_sequences)} 条序列")
    print(f"\n保存位置: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DPLM Potts 引导生成；默认较弱 beta 与 bias_clip 使生成贴近真实主体分布。")
    parser.add_argument("--num_samples", type=int, default=500, help="每组生成序列数")
    parser.add_argument("--max_iter", type=int, default=500, help="扩散解码迭代次数")
    parser.add_argument("--beta_naive", type=float, default=1.0, help="Naive-Potts 引导强度，越小越贴近 naive_real 主体（默认 0.3）")
    parser.add_argument("--beta_exper", type=float, default=1.0, help="Exper-Potts 引导强度，越小越贴近 exper_real 主体（默认 0.5）")
    parser.add_argument("--bias_clip", type=float, default=0.0, help="Potts 对 logits 偏置的裁剪上限，避免拉向离群点（默认 2.0，0 表示不裁剪）")
    parser.add_argument("--quick", action="store_true", help="快速测试: 2 条/组, 10 步")
    parser.add_argument("--uncond_only", action="store_true", help="仅生成 Uncond 序列")
    parser.add_argument("--output_suffix", type=str, default=None, help="输出文件名后缀，如 beta2 -> pr_naive_potts_beta2.fasta，避免覆盖")
    parser.add_argument("--potts_only", action="store_true", help="仅生成 Naive-Potts 与 Exper-Potts，跳过 Uncond")
    args = parser.parse_args()
    if args.quick:
        args.num_samples = 2
        args.max_iter = 10
        print(" quick 模式: num_samples=2, max_iter=10")
    main(
        num_samples=args.num_samples,
        max_iter=args.max_iter,
        beta_naive=args.beta_naive,
        beta_exper=args.beta_exper,
        bias_clip=args.bias_clip if args.bias_clip > 0 else None,
        uncond_only=args.uncond_only,
        output_suffix=args.output_suffix,
        potts_only=args.potts_only,
    )
