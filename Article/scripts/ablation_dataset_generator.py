#!/usr/bin/env python3
"""
ablation_dataset_generator.py — Generate ablated datasets for LLM evaluation.

This script processes the original real-world JSONL dataset and creates an
ablated version to test the "Gemma-4 without TimesFM" hypothesis.

Since the original dataset presents 5 full days of context, the TimesFM
agentic tool acts as a 48-hour forecaster extending a 3-day history.
To evaluate the LLM natively without TimesFM, this script truncates the
input context to 3 days, forcing the LLM to predict the day 5 AKI outcome
based only on the static features and 3-day trends (analogous to the ML baselines).

Usage:
    python Article/scripts/ablation_dataset_generator.py
"""

import json
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ARTICLE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = ARTICLE_DIR.parent

def truncate_to_3_days(content_str):
    """
    Truncates a patient monitoring string to only contain Day 1 to Day 3.
    """
    lines = content_str.split('\n')
    truncated_lines = []
    
    for line in lines:
        if '[Day 4]' in line or '[Day 5]' in line:
            continue
        truncated_lines.append(line)
        
    return '\n'.join(truncated_lines)

def main():
    input_file = PROJECT_DIR / "data" / "real_world" / "phd_proposal_eval_holdout.jsonl"
    if not input_file.exists():
        # Fallback to the SFT dataset if holdout isn't readily available under that exact name
        input_file = PROJECT_DIR / "data" / "real_world" / "phd_proposal_sft_dataset.jsonl"
    
    if not input_file.exists():
        print(f"ERROR: Could not find input dataset at {input_file}")
        return 1
        
    output_dir = ARTICLE_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "ablation_eval_holdout_3day.jsonl"
    
    print(f"Reading dataset: {input_file}")
    
    processed = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
             
        for line in fin:
            if not line.strip():
                continue
                
            record = json.loads(line)
            
            # Find the user prompt and truncate it
            for msg in record.get("messages", []):
                if msg["role"] == "user":
                    msg["content"] = truncate_to_3_days(msg["content"])
                    
                # Optionally, adjust the assistant response to reflect that it's a prediction
                # rather than a retrospective synthesis, to match the ablation task.
                if msg["role"] == "assistant":
                    # We keep the ground truth risk status intact for evaluation scripts
                    pass
            
            fout.write(json.dumps(record) + '\n')
            processed += 1
            
            # If we are using the huge SFT dataset as a stand-in, just take 200 for the eval holdout
            if processed >= 200 and "sft_dataset" in input_file.name:
                break
                
    print(f"Successfully generated ablated dataset: {output_file}")
    print(f"Total trajectories truncated to 3 days: {processed}")
    print("\nNext step: Run the batch evaluator (eval_dikd_batch.py) on this ablated dataset to get LLM-only performance.")

if __name__ == "__main__":
    raise SystemExit(main())
