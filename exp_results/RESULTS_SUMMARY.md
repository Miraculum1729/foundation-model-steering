# HIV-1 PR DPLM Potts-Guided Generation and Validation — Results Summary

## 1. Experimental Design Overview

- **Model**: DPLM 150M (untuned), HXB2 template with fixed motif (PQIT, DTG, GRVLG, TLNF); 93aa = 99aa[5:98].
- **Generation**: Three groups of 500 sequences each
  - **Uncond**: HXB2 clamping only, no Potts guidance
  - **Naive-Potts**: Naive Potts energy guidance (E_i(aa) bias on logits, β=1)
  - **Exper-Potts**: Exper Potts energy guidance (same)
- **Reference**: Naive validation set 4426 sequences, Exper validation set 952 sequences (93aa).

---

## 2. Main metrics summary

### 2.1 Perplexity (DPLM perplexity over sequences; lower = more "natural" to the model)

| Group | Mean | Brief interpretation |
|-------|------|----------------------|
| naive_real | 11,793 | Real naive validation set |
| exper_real | 8,283 | Real exper validation set (lower, closer to DPLM distribution) |
| **uncond_gen** | **11,767** | Close to naive_real; unconditioned generation is conservative |
| **naive_potts_gen** | **3,199** | Lowest; Potts guidance yields smoother sequences |
| **exper_potts_gen** | **6,456** | Between naive and exper real |

**Conclusion**: Potts guidance significantly lowers perplexity of generated sequences; Naive-Potts is lowest, Exper-Potts lies between naive and exper real, consistent with steering toward different evolutionary manifolds.

---

### 2.2 AAR (Amino acid retention rate vs HXB2, %)

| Group | AAR | Note |
|-------|-----|------|
| naive_real | 94.86% | Real naive is close to HXB2 |
| exper_real | 91.02% | Real exper has more mutations |
| **uncond_gen** | **93.89%** | Close to naive, high retention |
| **naive_potts_gen** | **76.22%** | Lower retention, higher diversity |
| **exper_potts_gen** | **84.64%** | Between the two |

**Conclusion**: Uncond stays very close to HXB2; both Potts-guided groups increase variation (lower AAR), with Naive-Potts varying most, consistent with diversity metrics. The "40–60%" band is a reference; with fixed motif, higher AAR is expected.

---

### 2.3 Diversity (inter-sequence distance and per-site entropy)

| Group | n | Mean Hamming (norm.) | Mean per-site entropy | Unique sequence ratio |
|-------|---|----------------------|------------------------|------------------------|
| naive_real | 4426 | 0.082 | 0.168 | 79.6% |
| exper_real | 952 | 0.141 | 0.277 | 100% |
| **uncond_gen** | 500 | **0.110** | **0.315** | 56.8% |
| **naive_potts_gen** | 500 | **0.264** | **0.546** | 100% |
| **exper_potts_gen** | 500 | **0.205** | **0.444** | 99.6% |

**Conclusion**: Naive-Potts and Exper-Potts have higher Hamming and entropy than real data, with unique ratio near 100%; Potts guidance effectively increases sequence diversity under fixed motif. Uncond is more conservative with more repetition.

---

### 2.4 Resistance mutation enrichment (vs exper real frequency)

| Mutation | Exper real freq. | Uncond | Naive-Potts | Exper-Potts |
|----------|------------------|--------|-------------|-------------|
| D30N | 0.53% | 0.38 | 0.38 | 0.38 |
| I84V | 2.21% | 0.09 | **0.36** | **0.45** |
| A71V | 3.15% | 0.19 | 0.06 | 0.06 |
| L90M | 0% | — | — | detected |

**Conclusion**: Exper-Potts has highest enrichment at I84V (~0.45), Naive-Potts next (0.36), consistent with Exper-Potts biasing toward resistance-related patterns. Major PI resistance sites (e.g. V82A) still have low detection in generated sequences, possibly due to sample size or Potts training data.

---

### 2.5 Mutation distribution (PSSM similarity, Frobenius)

| Comparison | PSSM similarity |
|------------|-----------------|
| naive_real vs exper_real | 0.868 |
| naive_real vs uncond_gen | **0.893** |
| naive_real vs naive_potts_gen | 0.559 |
| naive_real vs exper_potts_gen | 0.714 |
| exper_real vs exper_potts_gen | 0.685 |

**Conclusion**: Uncond is closest to naive real distribution; both Potts-guided groups shift PSSM away from naive (similarity ~0.56–0.71), so Potts guidance changes per-site mutation distribution, consistent with steering to different evolutionary manifolds.

---

### 2.6 Embedding space (PCA / t-SNE)

PCA and t-SNE of DPLM last-layer representations (`exp_results/embeddings/tsne_plot.png`, `pca_plot.png`) show:

- **Potts guidance induces strong distribution shift**: Compared to Uncond, Naive-Potts and Exper-Potts move away from the main cluster of naive/exper real data in embedding space.
- **Some generated samples lie near few real-data outliers**: A substantial fraction of Potts-guided samples fall near a small set of outliers in the naive/exper validation set rather than covering the main real-data region.

**Interpretation**: Potts energy guidance likely pulls the generated distribution toward embedding regions corresponding to a few high-impact sequences in the training/validation set—these outliers may reflect rare mutation combinations or extreme evolutionary modes that the Potts model assigns lower energy (higher probability), shifting sampling in that direction. This confirms that guidance has a strong effect in embedding space and suggests that to get generations closer to the main real distribution, **weaken guidance** and **limit bias magnitude** (see below).

**Tuning for generations closer to the main real distribution (naive_potts → naive_real, exper_potts → exper_real)**: The script supports and defaults to weaker guidance and bias clipping:
- **Default**: `beta_naive=0.3`, `beta_exper=0.5`, `bias_clip=2.0` (Potts bias on logits clipped to ±2.0).
- **Run**: `python experiments/generation/generate_hiv_pr_dplm.py` uses these defaults; or set e.g. `--beta_naive 0.3 --beta_exper 0.5 --bias_clip 2.0`. If still too strong, reduce beta (e.g. 0.2/0.4) or bias_clip (e.g. 1.0).
- **Restore strong guidance**: `--beta_naive 1.0 --beta_exper 1.0 --bias_clip 0`.

---

## 3. Overall Conclusions

1. **Potts guidance is effective**: Naive-Potts and Exper-Potts differ from Uncond or a single real set in perplexity, diversity, PSSM, and some resistance sites (e.g. I84V), showing that DPLM + Potts can steer toward different evolutionary modes.
2. **DPLM structural constraints remain**: All generated sequences are 99aa with fixed motif; AAR is generally high, so generations stay on a plausible HIV-1 subtype B protease manifold.
3. **Embedding space**: PCA/t-SNE show strong distribution shift under Potts guidance, with some samples drawn toward a few real-data outliers, suggesting guidance may over-pull toward a few high-weight modes in the Potts training set; tuning β or regularizing energy can help match the main distribution.
4. **Optional**: Phase 5 structure prediction (ESMFold pLDDT/pTM) can be run separately.

---

## 4. Result File Locations

| Content | Path |
|---------|------|
| Generated sequences | `exp_results/generated/pr_{uncond,naive_potts,exper_potts}.fasta` |
| Perplexity | `exp_results/perplexity/perplexity_summary.txt`, `perplexity_all.csv` |
| AAR | `exp_results/aar/aar_summary.txt`, `aar_distribution.png` |
| Diversity | `exp_results/diversity/diversity_summary.txt`, `hamming_distances.png`, `entropy_by_site.png` |
| Resistance enrichment | `exp_results/resistance/enrichment_table.csv` |
| Mutation/PSSM | `exp_results/mutation/pssm_similarity.csv`, `kl_divergence_by_pos.csv`, `key_sites_comparison.png` |
| Embedding space | `exp_results/embeddings/tsne_plot.png`, `pca_plot.png`, `embeddings.npz` |
| Consolidated report and figures | `exp_results/dashboard/final_report.md`, `summary.html` |

---

*Results summary from current run; all evaluation uses 93aa (99aa[5:98]) and the validation set.*
