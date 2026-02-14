# experiments 目录小结

本目录围绕 **HIV-1 PR 的 DPLM 生成与多维评估**，实现了数据准备、Potts 引导生成和 Phase 2–6 评估流程。

---

## 一、目录结构

```
experiments/
├── data_processing/     # 数据准备
├── generation/          # 序列生成
└── evaluation/          # 评估分析（Phase 2–6）
```

---

## 二、数据准备（data_processing）

### `prepare_hiv_data.py`

- **输入**：Stanford PR val 序列（naive/exper）、Mi3 训练的 Potts 模型 J 矩阵
- **处理**：
  - 将 Mi3 J 从 `(4278, 441)` 转为 `(L,L,q,q)`，构建 naive/exper 两套 Potts 模型
  - 计算 PSSM、Shannon 熵、突变频率等统计
  - 从 naive val set 提取 HXB2 参考序列
- **输出**：
  - `hiv_data/processed/potts_models.npz`：Potts 模型
  - `pr_naive_val.fasta`、`pr_exper_val.fasta`：验证集序列
  - `hiv_data/reference/hxb2_pr.fasta`：HXB2 参考
  - `pr_statistics.npz`：统计信息

---

## 三、序列生成（generation）

### `generate_hiv_pr_dplm.py`

- **模型**：DPLM 150M（airkingbd/dplm_150m）
- **模板**：99 aa HXB2，固定 motif（PQITL、DTG、GGIG、CTLNF），93aa = 99aa[5:98]
- **Potts 引导**：
  - `F_i(a) = h_i(a) + Σ_{j已解码} J_ij(a,x_j)`
  - `logits_mod = logits + beta * potts_bias`（负能量作为偏置）
- **生成模式**：
  - **Uncond**：仅 motif clamping，无 Potts
  - **Naive-Potts**：Naive Potts 引导
  - **Exper-Potts**：Exper Potts 引导
- **参数**：`beta`（引导强度）、`bias_clip`（偏置裁剪）
- **输出**：`pr_uncond.fasta`、`pr_naive_potts.fasta`、`pr_exper_potts.fasta` 等

### `run_phase2.py`

- Phase 2 占位脚本，调用 `generate_dplm.py` 做无条件生成

---

## 四、评估分析（evaluation）

### Phase 2–6 评估脚本

| 脚本 | Phase | 功能 | 输出 |
|------|-------|------|------|
| `calc_perplexity.py` | — | DPLM 对序列的 perplexity | `perplexity_summary.txt` |
| `calc_aar.py` | — | 相对 HXB2 的氨基酸保留率 | `aar_summary.txt` |
| `diversity_analysis.py` | 4.2 | Hamming 距离、位点熵、唯一序列比例 | `diversity_summary.txt`、熵图 |
| `mutation_analysis.py` | 3.3 | PSSM、关键位点 KL、突变分布比较 | `pssm_similarity.csv`、`key_sites_comparison.png` |
| `resistance_enrichment.py` | — | 按药物、Major/Minor 分组耐药突变 enrichment | `enrichment_by_drug_tier.csv` |
| `embedding_analysis.py` | 3.2 | DPLM embedding PCA/t-SNE 可视化 | `tsne_plot.png`、`pca_plot.png` |
| `embedding_uncond_vs_real.py` | — | Uncond vs Real 的嵌入对比 | 嵌入图 |
| `structure_quality.py` | 5.2 | ESMFold pLDDT/pTM（占位，未实际跑） | 占位输出 |
| `generate_report.py` | 6 | 综合报告与 Dashboard HTML | `final_report.md`、`summary.html` |

### 辅助脚本

- **`potts_mcmc_to_fasta.py`**：将 Mi3 Potts MCMC 采样序列转为 FASTA，供 embedding 分析

---

## 五、实验结果概况

| 指标 | naive_real | exper_real | uncond_gen | naive_potts_gen | exper_potts_gen |
|------|------------|------------|------------|-----------------|-----------------|
| Perplexity | ~11.8k | ~8.3k | ~11.8k | ~3.2k–6.9k | ~6.5k–8.2k |
| AAR | ~95% | ~91% | ~94% | ~76–90% | ~85–91% |
| 多样性（熵） | 0.17 | 0.28 | 0.11–0.32 | 0.22–0.55 | 0.21–0.44 |

- **Potts 引导**：显著降低 perplexity、提高多样性（AAR 下降、位点熵上升）
- **beta 调节**：beta=2 时易模式塌缩；推荐 beta≈0.3–0.5 + bias_clip 以贴近真实主体分布
- **耐药位点**：Exper-Potts 在 I84V 等位点 enrichment 更高

---

## 六、输出目录（exp_results）

```
exp_results/
├── generated/        # FASTA：uncond、naive/exper_potts、potts_mcmc
├── perplexity/       # perplexity_summary.txt、perplexity_all.csv
├── aar/              # aar_summary.txt、aar_distribution.png
├── diversity/        # diversity_summary.txt、entropy_by_site.png
├── mutation/         # pssm_similarity.csv、kl_divergence_by_pos.csv
├── resistance/       # enrichment_by_drug_tier.csv、enrichment_table_per_mutation.csv
├── embeddings/       # tsne_plot.png、pca_plot.png、embeddings.npz
├── dashboard/        # final_report.md、summary.html
└── RESULTS_SUMMARY.md
```

---

## 七、待核事项

- Potts 引导实现与 `naive_guidance.py`、discrete_flow_models 的一致性
- Mi3 J 的 reshape / 索引是否与 Mi3 原始格式一致
- `output_masks` 在 DPLM 中的含义
- Phase 5：ESMFold 结构预测（pLDDT/pTM）尚未实际运行
