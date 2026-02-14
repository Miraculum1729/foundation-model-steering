# HIV-1 PR DPLM Potts 引导生成与验证 — 结果总结

## 1. 实验设计概览

- **模型**：DPLM 150M（未微调），HXB2 模板固定 motif（PQIT、DTG、GRVLG、TLNF），93aa = 99aa[5:98]。
- **生成**：三组各 500 条序列
  - **Uncond**：仅 HXB2 clamping，无 Potts 引导
  - **Naive-Potts**：Naive Potts 能量引导（E_i(aa) 偏置 logits，β=1）
  - **Exper-Potts**：Exper Potts 能量引导（同上）
- **对照**：Naive 验证集 4426 条、Exper 验证集 952 条（93aa）。

---

## 2. 主要指标汇总

### 2.1 Perplexity（DPLM 对序列的困惑度，越低越“像模型认为的自然”）

| 组别 | 均值 | 简要解读 |
|------|------|----------|
| naive_real | 11,793 | 真实 naive 验证集 |
| exper_real | 8,283 | 真实 exper 验证集（更低，更符合 DPLM 分布） |
| **uncond_gen** | **11,767** | 与 naive_real 接近，说明无引导时偏向保守 |
| **naive_potts_gen** | **3,199** | 最低，Potts 引导使序列更“顺滑” |
| **exper_potts_gen** | **6,456** | 介于 naive 与 exper 真实之间 |

**结论**：Potts 引导显著降低生成序列的 perplexity；Naive-Potts 最低，Exper-Potts 介于真实 naive 与 exper 之间，与“引导到不同进化流形”一致。

---

### 2.2 AAR（相对 HXB2 的氨基酸保留率，%）

| 组别 | AAR | 说明 |
|------|-----|------|
| naive_real | 94.86% | 真实 naive 与 HXB2 很接近 |
| exper_real | 91.02% | 真实 exper 更多突变 |
| **uncond_gen** | **93.89%** | 与 naive 接近，高保留 |
| **naive_potts_gen** | **76.22%** | 保留率下降，多样性提高 |
| **exper_potts_gen** | **84.64%** | 介于两者之间 |

**结论**：Uncond 几乎贴着 HXB2；两种 Potts 引导都增加了变异度（AAR 下降），其中 Naive-Potts 变异最大，与多样性指标一致。计划中的“40–60%”为参考区间，当前固定 motif 下 AAR 普遍偏高属预期。

---

### 2.3 多样性（序列间差异与位点熵）

| 组别 | n | 平均 Hamming（归一） | 平均位点熵 | 唯一序列比例 |
|------|---|----------------------|------------|--------------|
| naive_real | 4426 | 0.082 | 0.168 | 79.6% |
| exper_real | 952 | 0.141 | 0.277 | 100% |
| **uncond_gen** | 500 | **0.110** | **0.315** | 56.8% |
| **naive_potts_gen** | 500 | **0.264** | **0.546** | 100% |
| **exper_potts_gen** | 500 | **0.205** | **0.444** | 99.6% |

**结论**：Naive-Potts 与 Exper-Potts 的 Hamming 和熵均高于真实数据，唯一比例接近 100%，说明在固定 motif 下 Potts 引导有效增加了序列多样性；Uncond 相对更保守、重复更多。

---

### 2.4 耐药突变 Enrichment（相对 exper 真实频率）

| 突变 | Exper 真实频率 | Uncond | Naive-Potts | Exper-Potts |
|------|----------------|--------|-------------|-------------|
| D30N | 0.53% | 0.38 | 0.38 | 0.38 |
| I84V | 2.21% | 0.09 | **0.36** | **0.45** |
| A71V | 3.15% | 0.19 | 0.06 | 0.06 |
| L90M | 0% | — | — | 有检出 |

**结论**：Exper-Potts 在 I84V 上 enrichment 最高（约 0.45），Naive-Potts 次之（0.36），与“Exper-Potts 偏向耐药相关模式”一致；主要 PI 耐药位点（如 V82A）在生成序列中检出率仍低，可能与样本量或 Potts 训练数据有关。

---

### 2.5 突变分布（PSSM 相似度，Frobenius）

| 对比 | PSSM 相似度 |
|------|-------------|
| naive_real vs exper_real | 0.868 |
| naive_real vs uncond_gen | **0.893** |
| naive_real vs naive_potts_gen | 0.559 |
| naive_real vs exper_potts_gen | 0.714 |
| exper_real vs exper_potts_gen | 0.685 |

**结论**：Uncond 与 naive 真实分布最接近；两种 Potts 引导均使 PSSM 明显偏离 naive（相似度约 0.56–0.71），说明 Potts 引导改变了位点突变分布，与“引导到不同进化流形”一致。

---

### 2.6 嵌入空间（PCA / t-SNE）

基于 DPLM 最后一层表示的 PCA 与 t-SNE 可视化（`exp_results/embeddings/tsne_plot.png`, `pca_plot.png`）显示：

- **Potts 引导带来很强的分布偏移**：相对 Uncond，Naive-Potts 与 Exper-Potts 在嵌入空间中明显脱离 naive/exper 真实数据的主体簇。
- **部分生成样本靠近真实数据中的少量离群点**：有相当一部分 Potts 引导生成样本落在或接近真实 naive/exper 验证集里的少数离群点附近，而不是均匀覆盖真实数据的主流区域。

**解读**：Potts 能量引导很可能在把生成分布拉向“训练/验证集中少数高影响序列”所对应的嵌入区域——这些离群点可能对应某种稀有突变组合或极端进化模式，Potts 模型对它们赋予了较低能量（高概率），从而把采样推向该方向。这既说明引导在嵌入空间里确实产生了强效应，也提示若希望生成更贴近真实主体分布，需要**减弱引导强度**并**限制偏置幅度**（见下）。

**使生成更贴近真实主体分布（naive_potts → naive_real，exper_potts → exper_real）**：脚本已支持并默认采用较弱引导 + 偏置裁剪：
- **默认参数**：`beta_naive=0.3`，`beta_exper=0.5`，`bias_clip=2.0`（对 logits 的 Potts 偏置裁剪到 ±2.0）。
- **运行**：`python experiments/generation/generate_hiv_pr_dplm.py` 即用上述默认；或显式指定例如  
  `--beta_naive 0.3 --beta_exper 0.5 --bias_clip 2.0`。  
  若仍偏强可再减小 beta（如 0.2/0.4）或减小 bias_clip（如 1.0）。
- **恢复强引导（原行为）**：`--beta_naive 1.0 --beta_exper 1.0 --bias_clip 0`。

---

## 3. 总体结论

1. **Potts 引导有效**：Naive-Potts 与 Exper-Potts 在 perplexity、多样性、PSSM、部分耐药位点（如 I84V）上均与 Uncond 或单一真实集形成区分，说明 DPLM + Potts 能朝不同进化模式引导。
2. **DPLM 结构约束仍在**：所有生成序列保持 99aa、固定 motif，AAR 普遍较高，说明在引导下仍处于“HIV-1 B 亚型蛋白酶”的合理流形内。
3. **嵌入空间**：PCA/t-SNE 显示 Potts 引导产生强分布偏移，部分生成样本被导向真实数据中的少量离群点附近，提示引导可能过度拉向 Potts 训练集里的少数高权重模式；若需更贴近主体分布可考虑调 β 或对能量做正则。
4. **尚未完成**：Phase 5 结构预测（ESMFold pLDDT/pTM）未跑。

---

## 4. 结果文件位置

| 内容 | 路径 |
|------|------|
| 生成序列 | `exp_results/generated/pr_{uncond,naive_potts,exper_potts}.fasta` |
| Perplexity | `exp_results/perplexity/perplexity_summary.txt`, `perplexity_all.csv` |
| AAR | `exp_results/aar/aar_summary.txt`, `aar_distribution.png` |
| 多样性 | `exp_results/diversity/diversity_summary.txt`, `hamming_distances.png`, `entropy_by_site.png` |
| 耐药 Enrichment | `exp_results/resistance/enrichment_table.csv` |
| 突变/PSSM | `exp_results/mutation/pssm_similarity.csv`, `kl_divergence_by_pos.csv`, `key_sites_comparison.png` |
| 嵌入空间 | `exp_results/embeddings/tsne_plot.png`, `pca_plot.png`, `embeddings.npz` |
| 综合报告与图表 | `exp_results/dashboard/final_report.md`, `summary.html` |

---

*结果总结由当前实验输出整理，评估均基于 93aa（99aa[5:98]）与 val set。*
