#!/usr/bin/env python3
"""Phase 5.2: Structure quality. Parse pLDDT/pTM after running cal_plddt_dir."""
from pathlib import Path
def main():
    base = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results/esmfold_predictions")
    base.mkdir(parents=True, exist_ok=True)
    (base / "pdb_quality_summary.csv").write_text("note\nPhase 5.1 not run.\n")
if __name__ == "__main__": main()
