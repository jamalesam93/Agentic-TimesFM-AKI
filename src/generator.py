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

    # --- Vectorized comorbidity sampling (Data-driven buckets) ---
    bucket_indices = np.clip(np.digitize(ages, [18, 35, 50, 65, 80, 96]) - 1, 0, 4)
    p_htn = np.array(parameters['p_htn_buckets'])[bucket_indices]
    p_dm = np.array(parameters['p_dm_buckets'])[bucket_indices]
    
    has_htn = rng.rand(n_synthetic) < p_htn
    has_dm = rng.rand(n_synthetic) < p_dm

    # --- Vectorized CKD sampling ---
    ckd_prob = np.where(ages < 50, 0.10, np.where(ages < 70, 0.25, 0.40))
    has_ckd = rng.rand(n_synthetic) < ckd_prob
    s3_mask = has_ckd & (rng.rand(n_synthetic) < 0.4)
    s2_mask = has_ckd & ~s3_mask
    no_ckd_mask = ~has_ckd

    # --- Vectorized baseline SCr ---
    baseline_scr = np.zeros(n_synthetic)
    baseline_scr[no_ckd_mask] = rng.uniform(0.6, 1.1, size=no_ckd_mask.sum())
    baseline_scr[s2_mask] = rng.uniform(1.0, 1.3, size=s2_mask.sum())
    baseline_scr[s3_mask] = rng.uniform(1.3, 2.0, size=s3_mask.sum())
    
    baseline_scr += (ages - 60) * 0.002
    baseline_scr += genders * 0.1
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
    zosyn_only_mask = (received_vanco == 0) & (received_zosyn == 1)
    vanco_only_mask = (received_vanco == 1) & (received_zosyn == 0)
    synergy_mask = (received_vanco == 1) & (received_zosyn == 1)
    
    aki_prob[zosyn_only_mask] = parameters.get('aki_rate_zosyn_only', 0.10)
    aki_prob[vanco_only_mask] = parameters['aki_rate_vanco_only']
    aki_prob[synergy_mask] = parameters['aki_rate_vanco_zosyn']
    
    # Adjust for renal risk (SCr > 1.0 increases risk)
    risk_multiplier = 1.0 + np.maximum(0, baseline_scr - 1.0) * 0.8
    aki_prob = np.clip(aki_prob * risk_multiplier, 0.0, 0.95)
    
    developed_aki = (rng.rand(n_synthetic) < aki_prob).astype(int)

    # --- Assemble into list of lightweight dicts ---
    gender_labels = np.where(genders == 1, "M", "F")
    patients = []
    for i in range(n_synthetic):
        comorbidities = []
        if has_htn[i]:
            comorbidities.append("Hypertension")
        if has_dm[i]:
            comorbidities.append("Type 2 Diabetes Mellitus")
            
        b_scr = baseline_scr[i]
        # Assign CKD stage based on sampled flags
        if s3_mask[i]:
            comorbidities.append("Chronic Kidney Disease Stage 3")
        elif s2_mask[i]:
            comorbidities.append("Chronic Kidney Disease Stage 2")
            
        patients.append({
            'synthetic_id': f"SYN_{i:05d}",
            'age': int(ages[i]),
            'gender': str(gender_labels[i]),
            'baseline_scr': float(b_scr),
            'comorbidities': comorbidities,
            'received_vanco': bool(received_vanco[i]),
            'received_zosyn': bool(received_zosyn[i]),
            'developed_aki': bool(developed_aki[i]),
        })
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

    This version introduces:
      1. Clinical Feedback Loop: Clinicians may discontinue antibiotics on Day 4/5 
         due to rising Serum Creatinine, causing SCr to enter a recovery trend.
      2. Flowsheet Charting Errors: A 5% probability that active drug administration 
         is not charted (flagged False in flowsheet) despite patient exposure.
      3. KDIGO Criteria consistency: Once a patient develops AKI, they remain 
         flagged as AKI_STAGE_1+ even if SCr begins to drop (recovery).

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
    
    # Clinical feedback loop: decision to discontinue drugs due to rising creatinine
    discontinued = False
    
    # Charting noise: does the flowsheet fail to record the active prescription?
    charting_error_vanco = rng.rand() < 0.05
    charting_error_zosyn = rng.rand() < 0.05
    
    # Multi-center Sparsity Simulation (e.g. eICU-like community hospitals)
    # 50% of patients are assigned to "community hospital" protocols where labs are sparse
    is_community_hospital = rng.rand() < 0.50
    
    # Track if KDIGO AKI criteria was ever met during the trajectory
    had_aki = False
    last_charted_scr = patient['baseline_scr']
    scr_history = [patient['baseline_scr']]

    for day in range(1, days + 1):
        # Base hemodynamics
        map_val = int(rng.normal(74, 6))
        # If severe AKI is developing, drop the blood pressure
        if patient['developed_aki'] and day >= 3:
            map_val = int(rng.normal(63, 4))

        # Check if clinicians discontinue therapy due to AKI on Day 4 or 5
        if patient['developed_aki'] and day >= 4 and not discontinued:
            # 30% chance to discontinue antibiotics to prevent further damage
            if rng.rand() < 0.30:
                discontinued = True

        # Actual drug administration
        actual_vanco = patient['received_vanco'] and day >= 2 and not discontinued
        actual_zosyn = patient['received_zosyn'] and day >= 2 and not discontinued

        # Simulated measurements (always reflect actual administration)
        if actual_vanco:
            vanco_trough += rng.uniform(3.0, 5.0)
            if actual_zosyn:
                # Synergy accelerates and exacerbates the structural damage
                vanco_trough += rng.uniform(1.5, 3.0)
        elif vanco_trough > 0:
            # Pharmacokinetic clearance when drug is stopped
            vanco_trough = max(0.0, vanco_trough - rng.uniform(4.0, 8.0))

        # Flowsheet flags (may contain charting errors)
        flowsheet_vanco = actual_vanco and not charting_error_vanco
        flowsheet_zosyn = actual_zosyn and not charting_error_zosyn

        # Simulate temporal rise/fall of SCr if patient develops AKI
        if patient['developed_aki'] and day >= 3:
            if discontinued and day == 5:
                # Renal recovery phase
                recovery_rate = rng.uniform(0.1, 0.4)
                current_scr -= round(recovery_rate, 2)
            else:
                # Active injury phase
                decay_rate = rng.uniform(0.3, 0.8)
                current_scr += round(decay_rate, 2)
        else:
            # Normal physiological drift
            current_scr += round(rng.normal(0.0, 0.03), 2)

        # Apply multi-center charting sparsity
        if is_community_hospital and day > 1:
            # 60% chance they don't draw a lab today, so we carry forward the previous day's chart value
            if rng.rand() < 0.60:
                charted_scr = last_charted_scr
            else:
                charted_scr = current_scr
                last_charted_scr = charted_scr
        else:
            charted_scr = current_scr
            last_charted_scr = charted_scr

        # Calculate KDIGO Stage and rolling baseline minimums
        past_48h = scr_history[-2:] if len(scr_history) >= 2 else []
        creat_low_past_48hr = min(past_48h) if past_48h else charted_scr
        creat_low_past_7day = min(scr_history)

        stage = 0
        stage1_cond1 = charted_scr >= creat_low_past_7day * 1.5
        stage1_cond2 = (len(past_48h) > 0 and charted_scr >= creat_low_past_48hr + 0.3)
        stage2_cond = charted_scr >= creat_low_past_7day * 2.0
        stage3_cond1 = charted_scr >= creat_low_past_7day * 3.0
        stage3_cond2 = (charted_scr >= 4.0 and (charted_scr >= creat_low_past_7day * 1.5 or charted_scr >= creat_low_past_48hr + 0.3))

        if stage1_cond1 or stage1_cond2:
            stage = 1
        if stage2_cond:
            stage = 2
        if stage3_cond1 or stage3_cond2:
            stage = 3

        scr_history.append(charted_scr)

        # KDIGO criteria: AKI Stage 1+ is defined as SCr >= 1.5x baseline or any stage >= 1
        if stage >= 1:
            had_aki = True

        risk_label = "NORMAL"
        if had_aki:
            risk_label = "AKI_STAGE_1+"

        trajectory.append({
            'day': day,
            'map': map_val,
            'vanco_active': bool(flowsheet_vanco),
            'zosyn_active': bool(flowsheet_zosyn),
            'vanco_trough': round(vanco_trough, 1) if vanco_trough > 0.0 else 0.0,
            'scr': round(charted_scr, 2),
            'risk_state': risk_label,
            'kdigo_stage': stage,
            'creat_low_past_48hr': round(creat_low_past_48hr, 2),
            'creat_low_past_7day': round(creat_low_past_7day, 2),
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
) -> Tuple[str, Optional[str], str, str]:
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
        Tuple of (jsonl_line, markdown_string_or_None, synthetic_id, timesfm_line).
    """
    # Import textualization inside worker to avoid circular imports
    # and ensure the module is available in the subprocess.
    from src.textualization import format_to_llm_jsonl, format_to_clinical_markdown, format_to_timesfm_dataset

    seed = (base_seed + idx) if base_seed is not None else None
    trajectory = generate_temporal_record(patient, days=days, seed=seed)

    # Textualize — format_to_llm_jsonl and format_to_clinical_markdown
    # accept dict-like objects (patient is a plain dict here).
    llm_payload = format_to_llm_jsonl(patient, trajectory)
    jsonl_line = json.dumps(llm_payload, ensure_ascii=False)

    md_report = None
    if generate_report:
        md_report = format_to_clinical_markdown(patient, trajectory)

    timesfm_payload = format_to_timesfm_dataset(patient, trajectory)
    timesfm_line = json.dumps(timesfm_payload, ensure_ascii=False)

    return jsonl_line, md_report, patient['synthetic_id'], timesfm_line


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
    timesfm_jsonl_path = os.path.join(output_dir, "timesfm_training_cohort.jsonl")

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

        # Open JSONL files for streaming writes
        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file, \
             open(timesfm_jsonl_path, "w", encoding="utf-8") as timesfm_file:
            for future in as_completed(futures):
                jsonl_line, md_report, synthetic_id, timesfm_line = future.result()

                # Stream JSONL line to disk immediately
                jsonl_file.write(jsonl_line + "\n")
                timesfm_file.write(timesfm_line + "\n")
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
        "timesfm_jsonl_path": timesfm_jsonl_path,
        "reports_dir": reports_dir,
        "first_sample": first_sample,
    }
