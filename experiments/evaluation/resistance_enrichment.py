#!/usr/bin/env python3
"""
Resistance mutation Enrichment analysis.
Statistics by drug and Major/Minor tier, based on Stanford HIV PI resistance mutation list.
Phase 2 convention: generated seqs 99 aa, 93aa = 99aa[5:98]; mutation positions 1-based PR(93aa).
Ref: MUTATIONS IN THE PROTEASE GENE ASSOCIATED WITH RESISTANCE TO PROTEASE INHIBITORS(PIs).md
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

# Phase 2: DPLM generates 99 aa, 93aa = 99aa[5:98]
DPLM_LEN = 99
PR_93_START, PR_93_END = 5, 98


def parse_mutation(s):
    """Parse mutation string, e.g. V32I -> (32, 'V', 'I', 'V32I')"""
    m = re.match(r"([A-Z])(\d+)([A-Z])", s.strip())
    if not m:
        return None
    wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
    return (pos, wt, mut, s.strip())


# PI resistance mutations: by drug, Major/Minor (from MUTATIONS IN THE PROTEASE GENE...md)
PI_MUTATIONS_BY_DRUG = {
    "Atazanavir": {
        "Major": ("V32I", "I50L", "I54V", "V82A", "I84V", "N88S"),
        "Minor": ("L10F", "L10I", "L10V", "K20T", "L24I", "L33F", "M46I", "M46L", "G48V", "F53L", "F53Y",
                  "I54L", "I54M", "I54T", "I54A", "I54S", "G73C", "G73S", "G73T", "G73A", "V82T", "V82F",
                  "V82L", "V82M", "V82S", "I85V", "L90M"),
    },
    "Darunavir": {
        "Major": ("I47V", "I50V", "I54L", "I54M", "L76V", "I84V"),
        "Minor": ("V11I", "V32I", "L33F", "T74P", "L89V"),
    },
    "Lopinavir": {
        "Major": ("V32I", "I47V", "I47A", "I50V", "I54V", "I54L", "I54M", "L76V", "V82A", "V82F", "V82T", "V82S", "I84V"),
        "Minor": ("L10F", "L10I", "L10R", "L10V", "K20M", "K20R", "L24I", "L33F", "M46I", "M46L", "F53L",
                  "I54A", "I54T", "I54S", "A71V", "A71T", "G73S", "L90M"),
    },
    "Tipranavir": {
        "Major": ("I47V", "Q58E", "T74P", "V82L", "V82T", "N83D", "I84V"),
        "Minor": ("L10V", "L33F", "M36I", "M36L", "M36V", "K43T", "M46L", "I54A", "I54M", "I54V", "H69K",
                  "H69R", "L89I", "L89M", "L89V"),
    },
    "Fosamprenavir": {
        "Major": ("I50V", "I84V"),
        "Minor": ("L10F", "L10I", "L10R", "L10V", "V32I", "M46I", "M46L", "I47V", "I54L", "I54V", "I54M",
                  "G73S", "L76V", "V82A", "V82F", "V82S", "V82T", "L90M"),
    },
    "Indinavir": {
        "Major": ("M46I", "M46L", "V82A", "V82F", "V82T", "I84V"),
        "Minor": ("L10I", "L10R", "L10V", "K20M", "K20R", "L24I", "V32I", "M36I", "I54V", "A71V", "A71T",
                  "G73S", "G73A", "L76V", "V77I", "L90M"),
    },
    "Nelfinavir": {
        "Major": ("D30N", "L90M"),
        "Minor": ("L10F", "L10I", "M36I", "M46I", "M46L", "A71V", "A71T", "V77I", "V82A", "V82F", "V82T",
                  "V82S", "I84V", "N88D", "N88S"),
    },
    "Saquinavir": {
        "Major": ("G48V", "L90M"),
        "Minor": ("L10I", "L10R", "L10V", "L24I", "I54V", "I54L", "I62V", "A71V", "A71T", "G73S", "V77I",
                  "V82A", "V82F", "V82T", "V82S", "I84V"),
    },
}

DRUGS = list(PI_MUTATIONS_BY_DRUG.keys())
TIERS = ["Major", "Minor"]


def build_mutation_list():
    """Build (pos, wt, mut, name, drug, tier) list from PI_MUTATIONS_BY_DRUG."""
    result = []
    for drug in DRUGS:
        for tier in TIERS:
            for s in PI_MUTATIONS_BY_DRUG[drug][tier]:
                parsed = parse_mutation(s)
                if parsed:
                    pos, wt, mut, name = parsed
                    result.append((pos, wt, mut, name, drug, tier))
    return result


def get_mutations_by_group():
    """Return {(drug, tier): [(pos, wt, mut, name), ...]}"""
    all_muts = build_mutation_list()
    by_group = defaultdict(list)
    for pos, wt, mut, name, drug, tier in all_muts:
        by_group[(drug, tier)].append((pos, wt, mut, name))
    return dict(by_group)


def get_unique_mutations():
    """Return deduplicated mutation list [(pos, wt, mut, name)] for per-mutation frequency."""
    seen = {}
    for pos, wt, mut, name, drug, tier in build_mutation_list():
        key = name
        if key not in seen:
            seen[key] = (pos, wt, mut, name)
    return list(seen.values())


def to_93aa_if_99(seq):
    """If 99 aa generated seq, take 93aa portion for PR 1-based mutation check."""
    if len(seq) == DPLM_LEN:
        return seq[PR_93_START:PR_93_END]
    return seq


def load_sequences(fasta_path):
    """Load FASTA sequences"""
    sequences = []
    names = []
    with open(fasta_path, 'r') as f:
        seq = ""
        name = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    names.append(name)
                name = line[1:]
                seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq)
            names.append(name)
    return names, sequences


def check_mutation(seq, position, wildtype, mutant):
    """Check if sequence contains a specific mutation."""
    if len(seq) < position:
        return False
    return seq[position - 1] == mutant


def calc_mutation_freq(sequences, mutation_list):
    """Compute mutation frequencies across sequences. mutation_list: [(pos, wt, mut, name), ...]"""
    freq = {mut[3]: 0.0 for mut in mutation_list}
    for seq in sequences:
        for pos, wt, mut, name in mutation_list:
            if check_mutation(seq, pos, wt, mut):
                freq[name] += 1.0
    n = len(sequences)
    for name in freq:
        freq[name] = freq[name] / n if n else 0
    return freq


def calc_group_freq(sequences, group_mutations):
    """Compute fraction of sequences with at least one mutation in the group.
    group_mutations: [(pos, wt, mut, name), ...]
    """
    if not sequences:
        return 0.0
    count = 0
    for seq in sequences:
        for pos, wt, mut, name in group_mutations:
            if check_mutation(seq, pos, wt, mut):
                count += 1
                break
    return count / len(sequences)


def main():
    data_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/hiv_data/processed")
    generated_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/generated")
    output_dir = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/resistance")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 3.4: Resistance mutation Enrichment (by drug, Major/Minor)")
    print("=" * 70)

    by_group = get_mutations_by_group()
    unique_muts = get_unique_mutations()
    print(f"\nTotal unique mutations: {len(unique_muts)}")
    print(f"Drug-tier groups: {len(by_group)}")

    # Load sequences
    _, naive_seqs = load_sequences(data_dir / "pr_naive_val.fasta")
    _, exper_seqs = load_sequences(data_dir / "pr_exper_val.fasta")
    _, gen_uncond_seqs = load_sequences(generated_dir / "pr_uncond.fasta")
    _, gen_naive_potts_seqs = load_sequences(generated_dir / "pr_naive_potts.fasta")
    _, gen_exper_potts_seqs = load_sequences(generated_dir / "pr_exper_potts.fasta")

    gen_uncond_seqs = [to_93aa_if_99(s) for s in gen_uncond_seqs]
    gen_naive_potts_seqs = [to_93aa_if_99(s) for s in gen_naive_potts_seqs]
    gen_exper_potts_seqs = [to_93aa_if_99(s) for s in gen_exper_potts_seqs]

    seq_groups = {
        "naive_real": naive_seqs,
        "exper_real": exper_seqs,
        "uncond_gen": gen_uncond_seqs,
        "naive_potts_gen": gen_naive_potts_seqs,
        "exper_potts_gen": gen_exper_potts_seqs,
    }

    # 1. Per-mutation frequency
    print("\n[1/3] Computing per-mutation frequency...")
    freq_by_group = {}
    for gname, seqs in seq_groups.items():
        freq_by_group[gname] = calc_mutation_freq(seqs, unique_muts)

    # 2. By (drug, tier): fraction of seqs with at least one mutation in group
    print("\n[2/3] By drug-Major/Minor group stats (fraction with any mutation)...")
    group_freq = defaultdict(dict)
    for (drug, tier), muts in sorted(by_group.items()):
        for gname, seqs in seq_groups.items():
            group_freq[(drug, tier)][gname] = calc_group_freq(seqs, muts)

    # 3. Compute Enrichment (generated groups relative to exper_real)
    print("\n[3/3] Computing Enrichment...")

    exper_freq = freq_by_group["exper_real"]

    # Save: per-mutation table (with drug/tier; same mutation may appear in multiple rows)
    mut_to_drug_tier = defaultdict(list)
    for pos, wt, mut, name, drug, tier in build_mutation_list():
        mut_to_drug_tier[name].append((drug, tier))

    rows_per_mut = []
    for pos, wt, mut, name in unique_muts:
        ef = exper_freq.get(name, 0)
        drugs_tiers = mut_to_drug_tier.get(name, [])
        drugs_str = "; ".join(f"{d}({t})" for d, t in drugs_tiers[:3])
        if len(drugs_tiers) > 3:
            drugs_str += f" +{len(drugs_tiers)-3}"
        def _enr(gen_f, exp_f):
            return (gen_f / exp_f) if exp_f > 0 else 0.0
        rows_per_mut.append({
            "mutation": name,
            "position": pos,
            "drugs_tiers": drugs_str,
            "naive_real_freq": freq_by_group["naive_real"][name],
            "exper_real_freq": ef,
            "uncond_gen_freq": freq_by_group["uncond_gen"][name],
            "naive_potts_gen_freq": freq_by_group["naive_potts_gen"][name],
            "exper_potts_gen_freq": freq_by_group["exper_potts_gen"][name],
            "uncond_enrichment": _enr(freq_by_group["uncond_gen"][name], ef),
            "naive_potts_enrichment": _enr(freq_by_group["naive_potts_gen"][name], ef),
            "exper_potts_enrichment": _enr(freq_by_group["exper_potts_gen"][name], ef),
        })

    with open(output_dir / "enrichment_table_per_mutation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "mutation", "position", "drugs_tiers", "naive_real_freq", "exper_real_freq",
            "uncond_gen_freq", "naive_potts_gen_freq", "exper_potts_gen_freq",
            "uncond_enrichment", "naive_potts_enrichment", "exper_potts_enrichment",
        ])
        w.writeheader()
        w.writerows(rows_per_mut)

    # Save: by (drug, tier) group summary
    def _enr(gen_frac, exp_frac):
        return (gen_frac / exp_frac) if exp_frac > 0 else 0.0

    rows_by_group = []
    for (drug, tier) in sorted(group_freq.keys()):
        gf = group_freq[(drug, tier)]
        ef = gf["exper_real"]
        r = {
            "drug": drug,
            "tier": tier,
            "n_mutations": len(by_group[(drug, tier)]),
            "naive_real_frac": gf["naive_real"],
            "exper_real_frac": ef,
            "uncond_gen_frac": gf["uncond_gen"],
            "naive_potts_gen_frac": gf["naive_potts_gen"],
            "exper_potts_gen_frac": gf["exper_potts_gen"],
            "uncond_enrichment": _enr(gf["uncond_gen"], ef),
            "naive_potts_enrichment": _enr(gf["naive_potts_gen"], ef),
            "exper_potts_enrichment": _enr(gf["exper_potts_gen"], ef),
        }
        rows_by_group.append(r)

    with open(output_dir / "enrichment_by_drug_tier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "drug", "tier", "n_mutations", "naive_real_frac", "exper_real_frac",
            "uncond_gen_frac", "naive_potts_gen_frac", "exper_potts_gen_frac",
            "uncond_enrichment", "naive_potts_enrichment", "exper_potts_enrichment",
        ])
        w.writeheader()
        w.writerows(rows_by_group)

    # Legacy: keep enrichment_table.csv (core mutation subset only)
    core_muts = ["D30N", "M46I", "M46L", "V82A", "I84V", "L90M", "L33F", "A71V", "A71T"]
    core_rows = [r for r in rows_per_mut if r["mutation"] in core_muts]
    with open(output_dir / "enrichment_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "mutation", "naive_real_freq", "exper_real_freq",
            "uncond_gen_freq", "naive_potts_gen_freq", "exper_potts_gen_freq",
            "uncond_enrichment", "naive_potts_enrichment", "exper_potts_enrichment",
        ])
        w.writeheader()
        for r in core_rows:
            w.writerow({k: r.get(k, 0) for k in w.fieldnames})

    # Console: by drug, Major/Minor
    print("\n" + "=" * 70)
    print("By drug, Major/Minor — fraction with any mutation & Enrichment")
    print("=" * 70)

    for drug in DRUGS:
        print(f"\n[{drug}]")
        for tier in TIERS:
            rr = next((r for r in rows_by_group if r["drug"] == drug and r["tier"] == tier), None)
            if rr is None:
                continue
            n = rr["n_mutations"]
            print(f"  {tier:5} (n={n:2}): naive={rr['naive_real_frac']:.1%} exper={rr['exper_real_frac']:.1%} | "
                  f"Uncond={rr['uncond_gen_frac']:.1%}({rr['uncond_enrichment']:.2f}) "
                  f"NaiveP={rr['naive_potts_gen_frac']:.1%}({rr['naive_potts_enrichment']:.2f}) "
                  f"ExperP={rr['exper_potts_gen_frac']:.1%}({rr['exper_potts_enrichment']:.2f})")

    print("\n" + "-" * 70)
    print("Key mutations (per-mutation Enrichment, Exper baseline>0 only):")
    print(f"{'Mutation':<8} {'Exper':>8} {'Uncond':>10} {'NaiveP':>10} {'ExperP':>10}")
    print("-" * 50)
    for r in rows_per_mut:
        if r["exper_real_freq"] > 0 and r["mutation"] in core_muts:
            print(f"{r['mutation']:<8} {r['exper_real_freq']:>8.3f} {r['uncond_enrichment']:>10.2f} "
                  f"{r['naive_potts_enrichment']:>10.2f} {r['exper_potts_enrichment']:>10.2f}")

    print("\nSaved:")
    print(f"  Per-mutation: {output_dir / 'enrichment_table_per_mutation.csv'}")
    print(f"  By drug-tier: {output_dir / 'enrichment_by_drug_tier.csv'}")
    print(f"  Legacy: {output_dir / 'enrichment_table.csv'}")
    print("\n" + "=" * 70)
    print("Phase 3.4 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
