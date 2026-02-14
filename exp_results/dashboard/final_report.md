# HIV-1 PR DPLM Validation Report

This report summarizes the generation evaluation of Foundation Model (DPLM) on HIV-1 protease. Motivation: DPLM is trained on massive data but still has limited generation capability in domain-specific settings (e.g., HIV PR resistance-related patterns). This experiment evaluates and visualizes from multiple angles: Perplexity, AAR, diversity, mutation distribution, resistance enrichment, and embedding space.

---

## Perplexity

```
naive_real_mean: 11792.718
exper_real_mean: 8282.878
uncond_gen_mean: 12254.112
naive_potts_gen_mean: 6861.896
exper_potts_gen_mean: 8242.356
```

## AAR

```
naive_real_mean: 0.9486
exper_real_mean: 0.9102
uncond_gen_mean: 0.9720
naive_potts_gen_mean: 0.8957
exper_potts_gen_mean: 0.9093
```

## Diversity

```
group	n	mean_hamming	mean_entropy	unique_ratio
naive_real	4426	0.082313	0.167644	0.795978
exper_real	952	0.140569	0.277341	1.000000
uncond_gen	500	0.047561	0.106039	0.508000
naive_potts_gen	500	0.113138	0.219076	0.992000
exper_potts_gen	500	0.104871	0.213379	0.972000
```
