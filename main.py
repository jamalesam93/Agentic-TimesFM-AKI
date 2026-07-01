import os
import json
import argparse
import pandas as pd
from src.data_extraction import generate_mock_historical_data, extract_statistical_parameters
from src.generator import synthesize_cohort, process_cohort_parallel

def main():
    parser = argparse.ArgumentParser(
        description="EHR Synthesis Engine - Vancomycin-Zosyn Synergistic Nephrotoxicity Generator"
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=500,
        help="Number of mock historical patients to simulate for parameter extraction (default: 500)"
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=10,
        help="Number of synthetic patient cohorts to generate (default: 10)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of temporal monitoring days per patient (default: 5)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Privacy budget (epsilon) for Differential Privacy. If omitted, standard extraction is used."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated datasets and reports (default: output)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--save-reports",
        type=int,
        default=5,
        help="Number of individual patient clinical markdown reports to save (default: 5)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for trajectory simulation. "
             "Defaults to os.cpu_count(). Set to 1 for sequential execution."
    )

    args = parser.parse_args()

    effective_workers = args.workers if args.workers is not None else os.cpu_count()

    print("=" * 70)
    print("               EHR SYNTHESIS PIPELINE INITIALIZATION")
    print("=" * 70)
    print(f"Configurations:")
    print(f"  - Historical database size: {args.n_patients} patients")
    print(f"  - Synthetic cohort size   : {args.n_synthetic} patients")
    print(f"  - ICU monitoring period   : {args.days} days")
    print(f"  - Differential Privacy    : Epsilon = {args.epsilon if args.epsilon is not None else 'DISABLED'}")
    print(f"  - Output directory        : {args.output_dir}")
    print(f"  - Seed                    : {args.seed}")
    print(f"  - Parallel workers        : {effective_workers}")
    print("-" * 70)

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Simulate historical EHR databases
    # -------------------------------------------------------------------------
    print("\n[Step 1/4] Simulating historical EHR databases (MIMIC-IV representation)...")
    raw_cohort = generate_mock_historical_data(n_patients=args.n_patients, seed=args.seed)
    raw_cohort_path = os.path.join(args.output_dir, "raw_historical_cohort.csv")
    raw_cohort.to_csv(raw_cohort_path, index=False)
    print(f"  -> Generated {len(raw_cohort)} raw historical trajectories.")
    print(f"  -> Saved raw records to {raw_cohort_path} (for verification/audit).")

    # -------------------------------------------------------------------------
    # Step 2: Parametric Extraction
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Extracting statistical parameters...")
    if args.epsilon is not None:
        print(f"  -> Applying Differential Privacy with Epsilon = {args.epsilon}")
    try:
        extracted_parameters = extract_statistical_parameters(raw_cohort, epsilon=args.epsilon)
    except ImportError as e:
        print(f"  [ERROR] {e}")
        print("  -> Falling back to standard parametric extraction without differential privacy.")
        extracted_parameters = extract_statistical_parameters(raw_cohort, epsilon=None)

    # Save extracted parameters as JSON
    params_path = os.path.join(args.output_dir, "extracted_parameters.json")
    # Save privacy budget ledger
    privacy_budget_path = os.path.join(args.output_dir, "privacy_budget.json")
    if "_privacy_ledger_structured" in extracted_parameters:
        ledger_data = {
            "differential_privacy_enabled": True,
            "total_epsilon": args.epsilon,
            "total_delta": 0.0,
            "mechanism": "Laplace Mechanism",
            "queries": extracted_parameters["_privacy_ledger_structured"]
        }
        # Remove the structured key from parameters so parameters JSON remains clean
        del extracted_parameters["_privacy_ledger_structured"]
    else:
        ledger_data = {
            "differential_privacy_enabled": False,
            "total_epsilon": None,
            "total_delta": None,
            "mechanism": None,
            "queries": []
        }
    with open(privacy_budget_path, "w", encoding="utf-8") as f:
        json.dump(ledger_data, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved formal privacy budget ledger to {privacy_budget_path}.")

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(extracted_parameters, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved extracted statistical soul to {params_path}.")
    print("  -> Key extracted metrics:")
    print(f"     * Mean Age: {extracted_parameters['age_mean']:.2f} yrs (Std: {extracted_parameters['age_std']:.2f})")
    print(f"     * Male Proportion: {extracted_parameters['male_proportion']*100:.1f}%")
    print(f"     * Synergy AKI Risk (Vanco + Zosyn): {extracted_parameters['aki_rate_vanco_zosyn']*100:.1f}%")
    print(f"     * Baseline AKI Risk: {extracted_parameters['aki_rate_baseline']*100:.1f}%")

    # -------------------------------------------------------------------------
    # Step 3: Vectorized Cohort Synthesis + Parallel Trajectory Pipeline
    # -------------------------------------------------------------------------
    print(f"\n[Step 3/4] Synthesizing {args.n_synthetic} patients & simulating trajectories...")
    print(f"  -> Vectorized baseline synthesis (numpy)...")
    patients = synthesize_cohort(
        extracted_parameters, n_synthetic=args.n_synthetic, seed=args.seed
    )

    # Save baseline cohort CSV
    synthetic_base_path = os.path.join(args.output_dir, "synthetic_cohort_baselines.csv")
    pd.DataFrame(patients).to_csv(synthetic_base_path, index=False)
    print(f"  -> Generated {len(patients)} baseline profiles -> {synthetic_base_path}")

    print(f"  -> Launching parallel trajectory pipeline ({effective_workers} workers)...")
    result = process_cohort_parallel(
        patients=patients,
        output_dir=args.output_dir,
        days=args.days,
        base_seed=args.seed,
        max_workers=args.workers,
        save_reports=args.save_reports,
        show_progress=True,
    )

    print(f"  -> Streamed {result['n_written']} JSONL records to {result['jsonl_path']}")
    print(f"  -> Streamed {result['n_written']} TimesFM records to {result['timesfm_jsonl_path']}")
    print(f"  -> Saved {result['n_reports']} clinical markdown reports to {result['reports_dir']}/")

    # -------------------------------------------------------------------------
    # Step 4: Completion & Sample Preview
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Pipeline successfully completed!")
    print("-" * 70)
    if result['first_sample'] is not None:
        print("  Sample LLM Fine-Tuning Payload:")
        print(json.dumps(result['first_sample'], indent=2))
    print("=" * 70)

if __name__ == "__main__":
    main()
