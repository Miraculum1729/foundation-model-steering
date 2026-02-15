# Experiments Directory Summary

This directory implements **HIV-1 PR DPLM generation and multi-dimensional evaluation**: data preparation, Potts-guided generation, and Phase 2–6 evaluation pipelines.

---

## 1. Directory Structure

```
experiments/
├── data_processing/     # Data preparation
├── generation/          # Sequence generation
└── evaluation/          # Evaluation and analysis (Phase 2–6)
```

---

## 2. Data Preparation (data_processing)

### `prepare_hiv_data.py`

- **Input**: Stanford PR validation sequences (naive/exper), Mi3-trained Potts model J matrices
- **Processing**:
  - Reshape Mi3 J from `(4278, 441)` to `(L,L,q,q)` and build naive/exper Potts models
  - Compute PSSM, Shannon entropy, mutation frequency, and related statistics
  - Extract HXB2 reference from naive validation set
- **Output**:
  - `hiv_data/processed/potts_models.npz`: Potts models
  - `pr_naive_val.fasta`, `pr_exper_val.fasta`: validation sequences
  - `hiv_data/reference/hxb2_pr.fasta`: HXB2 reference
  - `pr_statistics.npz`: statistics

---

## 3. Sequence Generation (generation)

### `generate_hiv_pr_dplm.py`

- **Model**: DPLM 150M (airkingbd/dplm_150m)
- **Template**: 99 aa HXB2 with fixed motif (PQITL, DTG, GGIG, CTLNF); 93aa = 99aa[5:98]
- **Potts guidance** (gauge-free, coupling J only):
  - `G_i(a) = Σ_{j decoded} J_ij(a,x_j)`; `logits_mod = logits + beta * potts_bias`
- **Generation modes**:
  - **Uncond**: motif clamping only, no Potts
  - **Naive-Potts**: Naive Potts guidance
  - **Exper-Potts**: Exper Potts guidance
- **Parameters**: `beta` (guidance strength), `bias_clip` (bias clipping)
- **Output**: `pr_uncond.fasta`, `pr_naive_potts.fasta`, `pr_exper_potts.fasta`, etc.

### `run_phase2.py`

- Phase 2 placeholder script; calls `generate_dplm.py` for unconditioned generation

---

## 4. Evaluation (evaluation)

### Phase 2–6 Evaluation Scripts

| Script | Phase | Function | Output |
|--------|-------|----------|--------|
| `calc_perplexity.py` | — | DPLM perplexity over sequences | `perplexity_summary.txt` |
| `calc_aar.py` | — | Amino acid retention rate vs HXB2 | `aar_summary.txt` |
| `diversity_analysis.py` | 4.2 | Hamming distance, per-site entropy, unique sequence ratio | `diversity_summary.txt`, entropy plots |
| `mutation_analysis.py` | 3.3 | PSSM, key-site KL, mutation distribution comparison | `pssm_similarity.csv`, `key_sites_comparison.png` |
| `resistance_enrichment.py` | — | PI resistance mutation enrichment by drug and Major/Minor tier | `enrichment_by_drug_tier.csv` |
| `embedding_analysis.py` | 3.2 | DPLM embedding PCA/t-SNE visualization | `tsne_plot.png`, `pca_plot.png` |
| `embedding_uncond_vs_real.py` | — | Uncond vs real embedding comparison | embedding plots |
| `structure_quality.py` | 5.2 | ESMFold pLDDT/pTM | CSV and summary |
| `generate_report.py` | 6 | Consolidated report and dashboard HTML | `final_report.md`, `summary.html` |

### Helper Scripts

- **`potts_mcmc_to_fasta.py`**: Convert Mi3 Potts MCMC sample sequences to FASTA for embedding analysis

---

## 5. Experimental Results Overview

| Metric | naive_real | exper_real | uncond_gen | naive_potts_gen | exper_potts_gen |
|--------|------------|------------|------------|-----------------|-----------------|
| Perplexity | ~11.8k | ~8.3k | ~11.8k | ~3.2k–6.9k | ~6.5k–8.2k |
| AAR | ~95% | ~91% | ~94% | ~76–90% | ~85–91% |
| Diversity (entropy) | 0.17 | 0.28 | 0.11–0.32 | 0.22–0.55 | 0.21–0.44 |

- **Potts guidance**: Significantly lowers perplexity and increases diversity (AAR down, per-site entropy up)
- **Beta tuning**: beta=2 can cause mode collapse; recommend beta≈0.3–0.5 + bias_clip to match real in-distribution
- **Resistance sites**: Exper-Potts shows higher enrichment at sites such as I84V

---

## 6. Output Directory (exp_results)

```
exp_results/
├── generated/        # FASTA: uncond, naive/exper_potts, potts_mcmc
├── perplexity/       # perplexity_summary.txt, perplexity_all.csv
├── aar/              # aar_summary.txt, aar_distribution.png
├── diversity/        # diversity_summary.txt, entropy_by_site.png
├── mutation/         # pssm_similarity.csv, kl_divergence_by_pos.csv
├── resistance/       # enrichment_by_drug_tier.csv, enrichment_table_per_mutation.csv
├── embeddings/       # tsne_plot.png, pca_plot.png, embeddings.npz
├── dashboard/        # final_report.md, summary.html
└── RESULTS_SUMMARY.md
```

---

## 7. Items to Verify

- Consistency of Potts guidance implementation with naive_guidance.py and discrete_flow_models
- Whether Mi3 J reshape/indexing matches Mi3 native format
- Meaning of `output_masks` in DPLM
- Phase 5: ESMFold structure prediction (pLDDT/pTM) can be run separately
