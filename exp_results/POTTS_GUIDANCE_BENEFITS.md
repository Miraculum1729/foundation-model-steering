# Benefits and Improvements from Potts Guidance (from current exp_results)

Based on the latest evaluation in `exp_results` (500 sequences per group), benefits of **Naive-Potts** and **Exper-Potts** over **Uncond** are summarized below.

---

## 1. Theory and implementation

- **Potts energy validation passed**  
  - Naive-Potts has **lower** mean energy than Uncond under the Naive Potts model; Exper-Potts has **lower** mean energy than Uncond under the Exper Potts model.  
  - Consistent with the paper (logits + λ·G favors lower energy); guidance direction and implementation are correct.

- **Sequences better match the target evolutionary manifold**  
  - Guided sequences have lower energy under the corresponding Potts model, i.e. they better match the co-evolution/statistical structure of the "naive" or "exper" population, supporting population-specific design or analysis.

---

## 2. Diversity improvement

| Metric | Uncond | Naive-Potts | Exper-Potts |
|--------|--------|-------------|-------------|
| Mean Hamming (norm.) | 0.041 | **0.052** | **0.060** |
| Mean per-site entropy | 0.085 | **0.104** | **0.127** |
| Unique sequence ratio | 50.4% | **71.0%** | **81.0%** |

- **Conclusion**: Potts guidance significantly increases sequence diversity and per-site variation; Exper-Potts is highest, Naive-Potts next, Uncond most conservative with most repetition.  
- **Benefit**: Under fixed HXB2 motif, more diverse candidate sequences are still obtained, supporting screening and broader sequence-space coverage.

---

## 3. Variation vs HXB2 (AAR)

| Group | AAR |
|-------|-----|
| naive_real | 94.86% |
| exper_real | 91.02% |
| Uncond | 97.49% |
| Naive-Potts | **96.54%** |
| Exper-Potts | **95.07%** |

- **Conclusion**: Uncond is closest to HXB2 (highest AAR); both Potts-guided groups lower AAR, closer to exper_real variation.  
- **Benefit**: Less over-conservatism toward a single reference; generated distribution closer to real "variant" populations, supporting exploration of non-consensus regions.

---

## 4. Mutation distribution closer to exper_real (selected sites)

**PSSM similarity (to exper_real)**  
- exper_real vs uncond_gen: 0.819  
- exper_real vs naive_potts_gen: 0.807  
- exper_real vs exper_potts_gen: 0.777  

Globally Potts groups have slightly lower PSSM similarity to exper_real, but **key-site KL(exper_real ‖ gen)** improves with Potts:

| Site | Uncond | Naive-Potts | Exper-Potts | Note |
|------|--------|-------------|-------------|------|
| 30 (D30N etc.) | 0.492 | **0.398** | 0.490 | Naive-Potts closer to exper |
| 82 (V82A etc.) | 0.051 | **0.034** | **0.036** | Both Potts groups closer to exper |
| Other key sites (46, 63, 71, 84, 90) | — | Same or slightly better | Same or slightly better | No degradation |

- **Conclusion**: At several resistance/key sites, Potts guidance brings generated per-site distribution closer to exper_real, favoring "exper-style" mutation patterns.  
- **Benefit**: Better suited for resistance- or experience-population sequence design or evaluation.

---

## 5. Resistance-related mutation fraction (selected drug–tier groups)

- **Tipranavir Minor** (e.g. I54V): exper 19.6%, uncond 7.4%, naive_potts 8.4%, **exper_potts 9.4%**  
  - Potts-guided groups have higher fraction of sequences with these mutations, closer to exper.  
- **Nelfinavir Minor**: exper 82.4%, uncond 92.4%, naive_potts 91.2%, **exper_potts 90.4%**  
  - Potts groups are slightly lower and closer to exper, avoiding excessive deviation from real distribution.  
- **Major resistance mutations** (e.g. I84V): fraction in generated sequences remains low for all three groups, consistent with current sample size and Potts training data; Potts does not introduce aberrant enrichment.

- **Benefit**: For some drug–tier groups, Potts guidance brings the fraction of resistance-related mutations in generated sequences closer to exper_real, supporting resistance-related analysis and design.

---

## 6. Structure quality (ESMFold)

| Group | mean pLDDT | mean pTM |
|-------|------------|----------|
| uncond | 84.22 | 0.838 |
| naive_potts | 84.02 | 0.837 |
| exper_potts | 83.91 | 0.836 |

- **Conclusion**: All three groups have similar pLDDT/pTM; Potts guidance does not harm overall folding or confidence.  
- **Benefit**: While improving diversity and matching exper, foldability remains comparable to Uncond.

---

## 7. Summary: benefits of Potts guidance

1. **Theory/implementation**: Guidance direction is correct; sequences have lower energy under the corresponding Potts model and better match the target population's statistical structure.  
2. **Diversity**: Hamming, per-site entropy, and unique ratio all increase; under fixed motif, more diverse candidates are still obtained.  
3. **Variation**: AAR is moderately reduced, reducing over-conservatism toward HXB2 and moving closer to variant populations.  
4. **Key sites**: At some resistance/key sites (e.g. 30, 82), per-site distribution is closer to exper_real (lower KL).  
5. **Resistance-related fraction**: For some drug–tier groups, the fraction of sequences with resistance-related mutations is closer to exper, supporting resistance analysis.  
6. **Structure**: pLDDT/pTM comparable to Uncond; no loss of structure quality.

**Note**: In the current run, DPLM perplexity differs little across the three groups (Uncond slightly lower); Potts benefits are mainly in diversity, AAR, key-site distribution, and resistance-related fraction. To get clearer perplexity separation, tune β / bias_clip or sampling settings and re-run.
