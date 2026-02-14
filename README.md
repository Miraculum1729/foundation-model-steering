# Foundation Model Steering — Domain-Specific Generation: Evaluation and Guidance

## Background and Motivation

Although Foundation Models (e.g., DPLM and other protein language models) are trained on massive sequence data and exhibit strong general-purpose representation capability, **they still show clear limitations in domain-specific sequence generation**:

- Difficulty capturing domain-specific evolutionary constraints (e.g., co-evolution patterns in HIV protease)
- Deviation from real data in fine-grained patterns such as drug-pressure-related mutations and resistance site distribution
- Unconditioned generation tends to be overly "conservative" or follow the "bulk distribution," lacking sufficient diversity or coverage of specific evolutionary manifolds

This repository focuses on **HIV-1 protease (PR)** as a concrete domain to study the generation limitations of Foundation Models. We explore **Potts model guidance** to steer the generation distribution toward different evolutionary modes (naive vs. drug-exposed exper), and evaluate and visualize results from **multiple perspectives**.

---

## Evaluation Dimensions and Visualization

We quantify and visualize Foundation Model generation quality along the following dimensions:

| Dimension | What We Assess | Limitations Revealed |
|-----------|----------------|----------------------|
| **Perplexity** | Model perplexity over sequences | Whether unconditioned generation is overly conservative; whether guided generation better matches domain distribution |
| **AAR** | Amino acid retention rate relative to reference (HXB2) | Whether generation is overly close to wild-type and lacks variation |
| **Diversity** | Inter-sequence Hamming distance, per-site Shannon entropy, unique sequence ratio | Whether there is mode collapse or excessive repetition |
| **Mutation Distribution** | PSSM, key-site KL divergence | Degree of deviation in site mutation patterns from real naive/exper |
| **Resistance Enrichment** | PI resistance mutations by drug and Major/Minor tier | Whether resistance-related mutation patterns are poorly covered |
| **Embedding Space** | PCA / t-SNE visualization of DPLM representations | Where generated samples lie in representation space (bulk vs. outlier regions of real data) |

---

## Directory Structure

| Directory | Description |
|-----------|-------------|
| **experiments/** | Data preparation, generation scripts, evaluation scripts (perplexity, AAR, diversity, mutation, resistance, embedding visualization, etc.) |
| **exp_results/** | Experimental results, figures, and dashboard report |
| **hiv_data/** | Processed PR sequences, Potts models, HXB2 reference |

---

## Main Findings

- **No Potts-guided (Uncond)**: Generation is close to naive real distribution but has limited diversity and poor coverage of resistance-related mutations
- **Potts-guided**: Significantly improves diversity and enrichment at some resistance sites (e.g., I84V); overly strong guidance can cause mode collapse or bias toward outlier modes
- **Embedding Space**: Potts guidance shifts generated samples away from the real bulk cluster in representation space, suggesting guidance strength should be tuned carefully

For more details, see [experiments/EXPERIMENTS_SUMMARY.md](experiments/EXPERIMENTS_SUMMARY.md) and [exp_results/RESULTS_SUMMARY.md](exp_results/RESULTS_SUMMARY.md).
