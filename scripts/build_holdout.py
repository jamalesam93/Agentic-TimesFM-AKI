#!/usr/bin/env python3
"""
Generate a held-out evaluation set for the DIKD AKI sentinel.

Uses the same data generation pipeline (data_extraction → generator → textualization)
with a different random seed to produce a separate evaluation cohort.

Usage:
  python scripts/build_holdout.py --n 200 --seed 9999 --out data/eval_holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data_extraction import generate_mock_historical_data, extract_statistical_parameters
from src.generator import synthesize_cohort, generate_temporal_record
from src.textualization import format_to_llm_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DIKD evaluation holdout set")
    parser.add_argument("--n", type=int, default=200, help="Number of patients (default: 200)")
    parser.add_argument("--seed", type=int, default=9999, help="Random seed (must differ from training)")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "data" / "eval_holdout.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    print(f"Generating {args.n} holdout patients (seed={args.seed})...")

    # Load pre-extracted parameters or generate them if missing
    params_path = PROJECT_DIR / "output" / "extracted_parameters.json"
    if params_path.exists():
        print(f"Loading clinical parameters from {params_path}")
        with params_path.open(encoding="utf-8") as f:
            params = json.load(f)
    else:
        print("Pre-extracted parameters not found. Extracting from simulated mock data...")
        raw_cohort = generate_mock_historical_data(n_patients=500, seed=42)
        params = extract_statistical_parameters(raw_cohort, epsilon=None)

    # Synthesize cohort baseline profiles
    patients = synthesize_cohort(params, n_synthetic=args.n, seed=args.seed)

    # Textualize and simulate temporal trajectories
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i, patient in enumerate(patients):
            # Per-patient seed derived from holdout seed
            traj_seed = args.seed + i
            traj = generate_temporal_record(patient, days=5, seed=traj_seed)
            record = format_to_llm_jsonl(patient, traj)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    # Count labels
    rows = [json.loads(line) for line in args.out.open(encoding="utf-8") if line.strip()]
    aki = sum(1 for r in rows if "AKI_STAGE_1+" in r["messages"][2]["content"])
    normal = count - aki

    print(f"\n[SUCCESS] Generated {count} holdout patients -> {args.out}")
    print(f"  AKI_STAGE_1+: {aki} ({100 * aki / count:.1f}%)")
    print(f"  NORMAL:       {normal} ({100 * normal / count:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

