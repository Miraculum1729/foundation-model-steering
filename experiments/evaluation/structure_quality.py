#!/usr/bin/env python3
"""
Phase 5: Structure quality via ESMFold.
Run ESMFold on generated sequences and aggregate pLDDT/pTM per group.

Before running ESMFold, activate the Python venv venv_esmfold:
  source /mnt/hbnas/home/pfp/hiv2026/dplm/venv_esmfold/bin/activate

Usage:
  python experiments/evaluation/structure_quality.py [--fasta-dir PATH] [--output-dir PATH] [--max-sequences N]
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Paths: __file__ = .../dplm/experiments/evaluation/structure_quality.py -> parents[2] = repo root dplm
DPLM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTA_DIR = DPLM_ROOT / "exp_results" / "generated"
DEFAULT_OUTPUT_DIR = DPLM_ROOT / "exp_results" / "esmfold_predictions"

FASTA_GROUPS = {
    "uncond": "pr_uncond.fasta",
    "naive_potts": "pr_naive_potts.fasta",
    "exper_potts": "pr_exper_potts.fasta",
}


def read_fasta(path):
    """Read FASTA, yield (header, sequence) pairs."""
    with open(path, "r") as f:
        seq, desc = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq and "X" not in seq:
                    yield desc, seq
                desc = line[1:]
                seq = ""
            else:
                seq += line
        if seq and "X" not in seq:
            yield desc, seq


def run_esmfold_and_save(
    fasta_path,
    output_dir,
    model_dir=None,
    max_tokens_per_batch=1024,
    max_sequences=None,
    cpu_only=False,
):
    """
    Run ESMFold on sequences in fasta_path, save PDBs and append pLDDT/pTM to CSV.
    Returns list of (name, mean_plddt, ptm).
    """
    try:
        import esm
        import torch
    except ImportError as e:
        print(
            "ESMFold import failed. Use: source .../venv_esmfold/bin/activate",
            file=sys.stderr,
        )
        return []

    if model_dir:
        import torch
        torch.hub.set_dir(model_dir)

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if cpu_only:
        model.esm.float()
        model.cpu()
        device = "cpu"
    else:
        model.cuda()
        device = "cuda"

    sequences = list(read_fasta(fasta_path))
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    results = []
    batch_headers, batch_seqs, num_tokens = [], [], 0

    for header, seq in sequences:
        if (num_tokens + len(seq) > max_tokens_per_batch) and batch_headers:
            # Run batch
            out = model.infer(batch_seqs, num_recycles=4)
            out = {k: v.cpu() for k, v in out.items()}
            for h, s, plddt, ptm in zip(
                batch_headers,
                batch_seqs,
                out["mean_plddt"].tolist(),
                out["ptm"].tolist(),
            ):
                plddt_val = float(plddt)
                ptm_val = float(ptm)
                results.append((h, plddt_val, ptm_val))
            batch_headers, batch_seqs, num_tokens = [], [], 0

        batch_headers.append(header)
        batch_seqs.append(seq)
        num_tokens += len(seq)

    if batch_headers:
        out = model.infer(batch_seqs, num_recycles=4)
        out = {k: v.cpu() for k, v in out.items()}
        for h, s, plddt, ptm in zip(
            batch_headers,
            batch_seqs,
            out["mean_plddt"].tolist(),
            out["ptm"].tolist(),
        ):
            results.append((h, float(plddt), float(ptm)))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fasta-dir",
        type=Path,
        default=DEFAULT_FASTA_DIR,
        help="Directory containing pr_*.fasta",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSV and logs",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Torch hub cache for ESM weights",
    )
    ap.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Limit sequences per group (for quick test)",
    )
    ap.add_argument(
        "--cpu-only",
        action="store_true",
        help="Use CPU only (slow)",
    )
    ap.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only aggregate existing pdb_quality_all.csv if present",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "pdb_quality_all.csv"
    summary_csv = args.output_dir / "pdb_quality_summary.csv"

    if args.skip_inference:
        if not out_csv.exists():
            print("No pdb_quality_all.csv found. Run without --skip-inference first.")
            sys.exit(1)
        rows = []
        with open(out_csv, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    else:
        all_rows = []
        for group_name, fname in FASTA_GROUPS.items():
            fp = args.fasta_dir / fname
            if not fp.exists():
                print(f"Skip {fp} (not found)")
                continue
            print(f"Running ESMFold on {fp}...")
            res = run_esmfold_and_save(
                fp,
                args.output_dir,
                model_dir=args.model_dir,
                max_sequences=args.max_sequences,
                cpu_only=args.cpu_only,
            )
            for name, plddt, ptm in res:
                all_rows.append({
                    "group": group_name,
                    "name": name,
                    "plddt": f"{plddt:.2f}",
                    "ptm": f"{ptm:.4f}",
                })
            print(f"  {group_name}: {len(res)} sequences")

        if not all_rows:
            print("No results. Writing placeholder.")
            (args.output_dir / "pdb_quality_all.csv").write_text(
                "group,name,plddt,ptm\n"
            )
            (args.output_dir / "pdb_quality_summary.csv").write_text(
                "note\nESMFold inference failed or no sequences.\n"
            )
            return

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group", "name", "plddt", "ptm"])
            w.writeheader()
            w.writerows(all_rows)
        rows = all_rows

    # Summary
    import numpy as np
    groups = {}
    for r in rows:
        g = r["group"]
        if g not in groups:
            groups[g] = {"plddt": [], "ptm": []}
        try:
            groups[g]["plddt"].append(float(r["plddt"]))
            groups[g]["ptm"].append(float(r["ptm"]))
        except (KeyError, ValueError):
            pass

    summary_lines = ["group,n,mean_plddt,std_plddt,mean_ptm,std_ptm"]
    for g in sorted(groups.keys()):
        plddt = np.array(groups[g]["plddt"])
        ptm = np.array(groups[g]["ptm"])
        n = len(plddt)
        if n > 0:
            summary_lines.append(
                f"{g},{n},{plddt.mean():.2f},{plddt.std():.2f},"
                f"{ptm.mean():.4f},{ptm.std():.4f}"
            )

    with open(summary_csv, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"Summary written to {summary_csv}")
    for line in summary_lines[1:]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
