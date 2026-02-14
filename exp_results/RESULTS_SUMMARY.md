# Experimental Results — Foundation Model Domain-Specific Generation Evaluation

## Evaluation Context

This experiment evaluates **DPLM generation on HIV-1 PR** from multiple angles to reveal Foundation Model limitations in domain-specific generation and the improvements and limitations of Potts guidance.

**Experimental design**:

- **Model**: DPLM 150M (unfinetuned), HXB2 template with fixed motif
- **Generation groups**: Uncond (no guidance), Naive-Potts, Exper-Potts; 500 sequences per group
- **Reference**: Naive validation set (4426 seqs), Exper validation set (952 seqs) as real data

---

## 1. Perplexity — Model "naturalness" of sequences

| Group | Mean | Interpretation |
|-------|------|----------------|
| naive_real | 11,793 | Real naive sequences |
| exper_real | 8,283 | Real exper sequences (better fit to DPLM distribution) |
| uncond_gen | 11,767 | Without guidance, close to naive, conservative |
| naive_potts_gen | 3,199 | Perplexity drops with guidance |
| exper_potts_gen | 6,456 | Between naive and exper |

**Conclusion**: Without guidance DPLM output is conservative; Potts guidance makes generation smoother but may deviate from bulk distribution.

---

## 2. AAR — Amino acid retention relative to reference

| Group | AAR | Interpretation |
|-------|-----|----------------|
| naive_real | 94.86% | Real naive close to HXB2 |
| exper_real | 91.02% | Real exper has more mutations |
| uncond_gen | 93.89% | Almost glued to HXB2, insufficient variation |
| naive_potts_gen | 76.22% | Higher variation with guidance |
| exper_potts_gen | 84.64% | Between the two |

**Conclusion**: Uncond generation is too conservative; Potts guidance increases variation toward exper-like diversity.

---

## 3. Diversity — Inter-sequence distance and per-site entropy

| Group | Mean Hamming (norm.) | Mean site entropy | Unique ratio |
|-------|----------------------|-------------------|--------------|
| naive_real | 0.082 | 0.168 | 79.6% |
| exper_real | 0.141 | 0.277 | 100% |
| uncond_gen | 0.110 | 0.315 | 56.8% |
| naive_potts_gen | 0.264 | 0.546 | 100% |
| exper_potts_gen | 0.205 | 0.444 | 99.6% |

**Conclusion**: Uncond unique ratio ~57%, clear repetition; Potts guidance brings unique ratio near 100% and increases Hamming/entropy, showing effective diversity gain.

---

## 4. Mutation distribution — PSSM similarity

| Comparison | PSSM similarity |
|------------|-----------------|
| naive_real vs uncond_gen | **0.893** (closest) |
| naive_real vs naive_potts_gen | 0.559 |
| naive_real vs exper_potts_gen | 0.714 |
| exper_real vs exper_potts_gen | 0.685 |

**Conclusion**: Uncond closest to naive real distribution; Potts guidance shifts mutation distribution away from naive toward exper-related patterns.

---

## 5. Resistance mutation enrichment

| Mutation | Uncond | Naive-Potts | Exper-Potts |
|----------|--------|-------------|-------------|
| I84V | 0.09 | 0.36 | **0.45** |
| D30N | 0.38 | 0.38 | 0.38 |

**Conclusion**: Exper-Potts has higher enrichment at I84V and other resistance sites, showing guidance steers toward resistance patterns; Uncond coverage is weak.

---

## 6. Embedding space (PCA / t-SNE)

- Potts guidance moves generated samples **away** from the real bulk cluster in embedding space
- Some generated samples cluster near **outliers** in real data, suggesting guidance may over-pull toward high-weight Potts modes
- To better match bulk distribution, reduce beta or use bias_clip

---

## 7. Overall conclusions

1. **Foundation Model limitations in domain-specific generation**: Overly conservative when unguided, low diversity, weak coverage of resistance-related mutations
2. **Potts guidance benefits**: Improves diversity, shifts mutation distribution, increases enrichment at some resistance sites
3. **Guidance limits**: Overly strong guidance can cause mode collapse or bias toward outlier modes; tune beta and bias_clip

---

## Result file locations

| Content | Path |
|---------|------|
| Generated sequences | `exp_results/generated/` |
| Perplexity | `exp_results/perplexity/` |
| AAR | `exp_results/aar/` |
| Diversity | `exp_results/diversity/` |
| Mutation/PSSM | `exp_results/mutation/` |
| Resistance enrichment | `exp_results/resistance/` |
| Embedding visualization | `exp_results/embeddings/` |
| Summary report | `exp_results/dashboard/` |
