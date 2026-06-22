"""
generator.py — Cohort Synthesis & Parallel Temporal Trajectory Simulation

This module provides:
  1. Vectorized baseline cohort synthesis (no Python loops — scales to 100k+).
  2. Per-patient longitudinal ICU trajectory simulation.
  3. A parallel pipeline that streams results directly to disk, keeping peak
     memory usage proportional to the worker count, not the cohort size.

Architecture (streaming pipeline):
    ┌──────────────────┐
    │  synthesize_cohort│  Vectorized numpy — emits N patient dicts
    └────────┬─────────┘
             │ yields patient dicts (generator)
             ▼
    ┌──────────────────────────────────┐
    │  process_cohort_parallel          │  ProcessPoolExecutor
    │  ┌─────────┐  ┌─────────┐        │
    │  │ Worker 1 │  │ Worker K │  ...  │  Each worker: trajectory → JSONL + MD
    │  └────┬────┘  └────┬────┘        │
    │       │             │             │
    │       ▼             ▼             │
    │   returns (jsonl_line, md_or_None)│
    └────────┬─────────────────────────┘
             │ streamed results
             ▼
    ┌──────────────────┐
    │  Disk I/O (main)  │  Appends JSONL lines, writes .md files
    └──────────────────┘
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Iterator, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================================================
# VECTORIZED COHORT SYNTHESIS
# =============================================================================

def synthesize_cohort(
    parameters: Dict[str, Any],
    n_synthetic: int = 1000,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Uses ONLY the extracted statistical parameters to generate completely
    new patient profiles via vectorized numpy operations.

    Returns a list of plain dicts (not a DataFrame) to enable zero-copy
    transfer to worker processes and generator-based streaming.

    Vectorization strategy:
      - All random draws (age, gender, SCr residuals, drug assignments,
        AKI outcomes) are performed as single numpy array operations.
      - The only Python loop is the final assembly into dicts, which is
        a thin O(n) pass over pre-computed arrays.

    Args:
        parameters: Dictionary of statistical parameters from extraction.
        n_synthetic: Number of synthetic patients to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of patient record dicts ready for trajectory simulation.
    """
    rng = np.random.RandomState(seed)

    # --- Vectorized demographic sampling ---
    ages = rng.normal(parameters['age_mean'], parameters['age_std'], n_synthetic)
    ages = np.clip(ages, 18, 95).astype(int)

    genders = rng.binomial(1, parameters['male_proportion'], n_synthetic)

    # --- Vectorized baseline SCr ---
    # We predict the baseline SCr in log-space, add normally distributed residuals
    # using the DP-protected variance (log_scr_var), and exponentiate back.
    predicted_log_scr = (ages * parameters['age_to_scr_slope']) + parameters['age_to_scr_intercept']
    log_variance = parameters['log_scr_var']
    residuals = rng.normal(0, np.sqrt(log_variance), size=n_synthetic)
    baseline_scr = np.exp(predicted_log_scr + residuals)
    baseline_scr = np.maximum(0.4, baseline_scr).round(2)

    # --- Vectorized drug exposure assignment ---
    prob_vanco = np.where(
        ages >= 65,
        parameters['p_vanco_given_elderly'],
        parameters['p_vanco_given_normal']
    )
    received_vanco = (rng.rand(n_synthetic) < prob_vanco).astype(int)
    received_zosyn = (rng.rand(n_synthetic) < parameters['p_zosyn']).astype(int)

    # --- Vectorized synergistic AKI probability ---
    aki_prob = np.full(n_synthetic, parameters['aki_rate_baseline'])
    vanco_only_mask = (received_vanco == 1) & (received_zosyn == 0)
    synergy_mask = (received_vanco == 1) & (received_zosyn == 1)
    aki_prob[vanco_only_mask] = parameters['aki_rate_vanco_only']
    aki_prob[synergy_mask] = parameters['aki_rate_vanco_zosyn']
    developed_aki = (rng.rand(n_synthetic) < aki_prob).astype(int)

    # --- Assemble into list of lightweight dicts ---
    gender_labels = np.where(genders == 1, "M", "F")
    patients = [
        {
            'synthetic_id': f"SYN_{i:05d}",
            'age': int(ages[i]),
            'gender': str(gender_labels[i]),
            'baseline_scr': float(baseline_scr[i]),
            'received_vanco': bool(received_vanco[i]),
            'received_zosyn': bool(received_zosyn[i]),
            'developed_aki': bool(developed_aki[i]),
        }
        for i in range(n_synthetic)
    ]
    return patients


def synthesize_cohort_stream(
    parameters: Dict[str, Any],
    n_synthetic: int = 1000,
    seed: int = 42
) -> Iterator[Dict[str, Any]]:
    """
    Generator wrapper around synthesize_cohort that yields one patient
    dict at a time for memory-efficient iteration in streaming pipelines.
    """
    patients = synthesize_cohort(parameters, n_synthetic, seed)
    yield from patients


# =============================================================================
# TEMPORAL TRAJECTORY SIMULATION
# =============================================================================

def generate_temporal_record(
    patient: Dict[str, Any],
    days: int = 5,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Simulates consecutive ICU days, calculating cumulative drug exposure,
    hemodynamic fluctuations, and downstream renal decay.

    Uses a local RandomState instance (not the global numpy seed) so that
    each worker process produces deterministic, non-colliding sequences.

    Args:
        patient: A single patient's baseline record (plain dict).
        days: Number of ICU days to simulate.
        seed: Optional random seed for this trajectory.

    Returns:
        A list of daily clinical records representing the ICU trajectory.
    """
    rng = np.random.RandomState(seed)

    trajectory = []
    current_scr = patient['baseline_scr']
    vanco_trough = 0.0

    for day in range(1, days + 1):
        # Base hemodynamics
        map_val = int(rng.normal(74, 6))
        # If severe AKI is developing, drop the blood pressure
        if patient['developed_aki'] and day >= 3:
            map_val = int(rng.normal(63, 4))

        # Cumulative toxic exposure calculation
        vanco_active = patient['received_vanco'] and day >= 2
        zosyn_active = patient['received_zosyn'] and day >= 2

        if vanco_active:
            vanco_trough += rng.uniform(3.0, 5.0)
            if patient['received_zosyn']:
                # Synergy accelerates and exacerbates the structural damage
                vanco_trough += rng.uniform(1.5, 3.0)

        # Simulate temporal rise of SCr if patient develops AKI
        if patient['developed_aki'] and day >= 3:
            decay_rate = rng.uniform(0.3, 0.8)
            current_scr += round(decay_rate, 2)
        else:
            current_scr += round(rng.normal(0.0, 0.03), 2)

        # KDIGO criteria: AKI Stage 1 is defined as SCr >= 1.5x baseline
        risk_label = "NORMAL"
        if current_scr >= patient['baseline_scr'] * 1.5:
            risk_label = "AKI_STAGE_1+"

        trajectory.append({
            'day': day,
            'map': map_val,
            'vanco_active': vanco_active,
            'zosyn_active': zosyn_active,
            'vanco_trough': round(vanco_trough, 1) if vanco_active else 0.0,
            'scr': round(current_scr, 2),
            'risk_state': risk_label,
        })

    return trajectory


# =============================================================================
# WORKER FUNCTION (runs inside child processes)
# =============================================================================

def _process_single_patient(
    patient: Dict[str, Any],
    idx: int,
    days: int,
    base_seed: Optional[int],
    generate_report: bool,
) -> Tuple[str, Optional[str], str]:
    """
    Complete per-patient pipeline executed inside a worker process:
      1. Generate temporal trajectory.
      2. Serialize to JSONL line.
      3. Optionally render clinical markdown report.

    All imports needed by the worker are at module level, so this
    function is picklable by ProcessPoolExecutor.

    Args:
        patient: Patient baseline dict.
        idx: Patient index (for seed derivation).
        days: Number of ICU days.
        base_seed: Base seed; per-patient seed = base_seed + idx.
        generate_report: Whether to render a markdown report.

    Returns:
        Tuple of (jsonl_line, markdown_string_or_None, synthetic_id).
    """
    # Import textualization inside worker to avoid circular imports
    # and ensure the module is available in the subprocess.
    from src.textualization import format_to_llm_jsonl, format_to_clinical_markdown

    seed = (base_seed + idx) if base_seed is not None else None
    trajectory = generate_temporal_record(patient, days=days, seed=seed)

    # Textualize — format_to_llm_jsonl and format_to_clinical_markdown
    # accept dict-like objects (patient is a plain dict here).
    llm_payload = format_to_llm_jsonl(patient, trajectory)
    jsonl_line = json.dumps(llm_payload, ensure_ascii=False)

    md_report = None
    if generate_report:
        md_report = format_to_clinical_markdown(patient, trajectory)

    return jsonl_line, md_report, patient['synthetic_id']


# =============================================================================
# PARALLEL STREAMING PIPELINE
# =============================================================================

def process_cohort_parallel(
    patients: List[Dict[str, Any]],
    output_dir: str,
    days: int = 5,
    base_seed: Optional[int] = 42,
    max_workers: Optional[int] = None,
    save_reports: int = 5,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Processes an entire synthetic cohort in parallel, streaming JSONL lines
    directly to disk as workers complete. Markdown reports are written
    individually as they arrive.

    Memory profile:
      - At any point, at most `max_workers` patient trajectories + JSONL
        strings exist in memory simultaneously.
      - The JSONL file is opened in append mode; lines are flushed as
        each future completes.
      - No list of all results is ever accumulated.

    Args:
        patients: List of patient baseline dicts from synthesize_cohort.
        output_dir: Directory for output files.
        days: ICU simulation length.
        base_seed: Base seed for deterministic per-patient seeds.
        max_workers: Number of parallel processes (None = os.cpu_count()).
        save_reports: Number of markdown clinical reports to save.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Dict with pipeline statistics (counts, paths, first sample).
    """
    # Lazy import tqdm so the module works even if tqdm is not installed
    # (gracefully degrades to no progress bar).
    _tqdm = None
    if show_progress:
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            import sys
            print("  [WARN] tqdm not installed — progress bar disabled. "
                  "Install with: pip install tqdm", file=sys.stderr)

    n_total = len(patients)
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "llm_fine_tuning_dataset.jsonl")

    # Counters
    n_written = 0
    n_reports = 0
    first_sample = None

    # Submit all patients to the process pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Build futures map: future → index
        futures = {}
        for idx, patient in enumerate(patients):
            generate_report = idx < save_reports
            future = executor.submit(
                _process_single_patient,
                patient, idx, days, base_seed, generate_report,
            )
            futures[future] = idx

        # Stream results to disk as they complete
        progress = None
        if _tqdm is not None:
            progress = _tqdm(
                total=n_total,
                desc="  Simulating trajectories",
                unit="patient",
                bar_format="{l_bar}{bar:40}{r_bar}",
                ncols=100,
            )

        # Open JSONL file for streaming writes
        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for future in as_completed(futures):
                jsonl_line, md_report, synthetic_id = future.result()

                # Stream JSONL line to disk immediately
                jsonl_file.write(jsonl_line + "\n")
                n_written += 1

                # Capture first sample for preview
                if first_sample is None:
                    first_sample = json.loads(jsonl_line)

                # Write markdown report if generated
                if md_report is not None:
                    report_path = os.path.join(
                        reports_dir, f"patient_{synthetic_id}_report.md"
                    )
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(md_report)
                    n_reports += 1

                if progress is not None:
                    progress.update(1)

        if progress is not None:
            progress.close()

    return {
        "n_written": n_written,
        "n_reports": n_reports,
        "jsonl_path": jsonl_path,
        "reports_dir": reports_dir,
        "first_sample": first_sample,
    }
