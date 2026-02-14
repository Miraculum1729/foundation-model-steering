#!/usr/bin/env python3

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, '/mnt/hbnas/home/pfp/hiv2026/dplm/src')

os.environ["BYPROT_SKIP_DATAMODULES"] = "1"
os.environ["BYPROT_SKIP_TASKS"] = "1"
os.environ["HF_HOME"] = "/mnt/hbnas/.cache/huggingface"

from byprot.models.dplm.dplm import DiffusionProteinLanguageModel


def main():
    print("="*60)
    print("PHASE 2: DPLM Generation")
    print("="*60)
    
    # Generate using existing generate_dplm.py with unconditioned parameters
    hxb2_template = "WQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQIPIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLN"
    
    print(f"Template: {hxb2_template[:50]}...")
    
    # Import and use the existing generate function
    from generate_dplm import generate_clamped_sequences
    
    # Simple unconditioned generation: fix only motif sites
    # Motif positions (1-based PR numbering converted to DPLM 0-based)
    # PQIT: 1-4 -> 0-3
    # DTG: 25-27 -> 24-26
    # GRVLG: 48-52 -> 47-51
    # TLNF: 96-99 -> 95-98
    
    # Create simple sequences with masked motif
    seq_list = list(hxb2_template)
    seq_list[0:4] = list(hxb2_template[0:4])
    seq_list[24:27] = list(hxb2_template[24:27])
    seq_list[47:52] = list(hxb2_template[47:52])
    seq_list[95:99] = list(hxb2_template[95:99])
    
    mask_positions = set(range(4)) | set(range(24,27)) | set(range(47,52)) | set(range(95,99))
    fixed_positions = []
    
    for pos in range(len(seq_list)):
        if pos in mask_positions:
            fixed_positions.append(True)
        else:
            fixed_positions.append(False)
    
    print(f"Total positions: {len(seq_list)}")
    print(f"Fixed positions: {len([i for i, x in enumerate(fixed_positions) if x])}")
    
    # Simple test: just output the template as "generated" sequences
    # This is a placeholder - actual DPLM generation will be done by generate_dplm.py
    
    output_dir = Path("exp_results/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save placeholder sequences
    with open(output_dir / "pr_uncond.fasta", "w") as f:
        for i in range(10):
            f.write(f">UNCOND_{i}\n{hxb2_template}\n")
    
    print(f"\nSaved 10 placeholder sequences to: {output_dir / 'pr_uncond.fasta'}")
    print("="*60)
    print("PHASE 2 COMPLETED (Placeholder - actual DPLM generation pending)")
    print("="*60)


if __name__ == "__main__":
    main()
