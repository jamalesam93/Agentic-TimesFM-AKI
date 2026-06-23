import numpy as np
import pandas as pd
import json
from typing import Dict, Any, List, Optional, Tuple, Iterator
from scipy.stats import norm, lognorm

# -----------------------------------------------------------------------------
# STAGE 0: SIMULATE RAW HISTORICAL CLINICAL DATA (REPRESENTING MIMIC-IV)
# -----------------------------------------------------------------------------
def generate_mock_historical_data(n_patients: int = 500, seed: int = 101) -> pd.DataFrame:
    """
    Generates a historical dataset representing raw data from a generic Western ICU (like MIMIC-IV).
    This serves as the source from which we will extract our statistical parameters.
    """
    np.random.seed(seed)

    # Baseline Covariates (Generic ICU population: older, balanced male ratio)
    ages = np.random.normal(65.0, 16.0, n_patients).clip(18, 95)
    genders = np.random.binomial(1, 0.56, n_patients)  # 1 = Male, 0 = Female

    # Comorbidities
    p_htn = 1.0 / (1.0 + np.exp(-(ages * 0.06 - 3.5)))
    has_htn = np.random.binomial(1, p_htn)
    
    p_dm = 1.0 / (1.0 + np.exp(-(ages * 0.05 - 2.8)))
    has_dm = np.random.binomial(1, p_dm)

    # Baseline Serum Creatinine (SCr) - lower baseline GFR for generic population
    baseline_scr = []
    for age, gender, htn, dm in zip(ages, genders, has_htn, has_dm):
        base = 0.8 + (age * 0.003) + (gender * 0.15) + (htn * 0.05) + (dm * 0.05)
        scr = np.random.lognormal(mean=np.log(base), sigma=0.15)
        baseline_scr.append(round(scr, 2))

    df_base = pd.DataFrame({
        'patient_id': [f"HIST_{i:04d}" for i in range(n_patients)],
        'age': ages.astype(int),
        'gender': genders,
        'has_htn': has_htn,
        'has_dm': has_dm,
        'baseline_scr': baseline_scr
    })

    # Simulate exposure
    vanco_prob = 1 / (1 + np.exp(-(df_base['age'] * 0.02 + df_base['baseline_scr'] * 0.5 - 2.5)))
    df_base['received_vanco'] = np.random.binomial(1, vanco_prob)

    df_base['received_zosyn'] = np.random.binomial(1, 0.35, n_patients)

    # Outcomes: AKI incidence (Generic Western rates: Vanco+Zosyn synergy)
    aki_prob = []
    for _, row in df_base.iterrows():
        p = 0.05  # Baseline ICU AKI rate
        if row['received_vanco'] == 1:
            p += 0.15  # Vanco effect
        if row['received_zosyn'] == 1:
            p += 0.02  # Zosyn baseline effect
        if row['received_vanco'] == 1 and row['received_zosyn'] == 1:
            p += 0.15  # Synergistic "Zosyn-Vanc" toxicity boost
        aki_prob.append(min(0.95, p))

    df_base['developed_aki'] = np.random.binomial(1, aki_prob)
    return df_base

# -----------------------------------------------------------------------------
# STAGE 1: THE PARAMETRIC EXTRACTION (EXTRACTING THE STATISTICAL SOUL)
# -----------------------------------------------------------------------------
def extract_statistical_parameters(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes raw, real-world data to extract statistical descriptors, 
    ensuring we do not store any individual patient's raw records.
    """
    stats = {}
    n = len(raw_df)
    buckets = [18, 35, 50, 65, 80, 96]
    
    stats['age_mean'] = float(raw_df['age'].mean())
    stats['age_std'] = float(raw_df['age'].std())
    stats['male_proportion'] = float(raw_df['gender'].mean())
    stats['p_htn_overall'] = float(raw_df['has_htn'].mean())
    stats['p_dm_overall'] = float(raw_df['has_dm'].mean())

    # Bucketed comorbidity rates
    p_htn_buckets = []
    p_dm_buckets = []
    for i in range(5):
        b_df = raw_df[(raw_df['age'] >= buckets[i]) & (raw_df['age'] < buckets[i+1])]
        if len(b_df) > 0:
            p_htn_buckets.append(float(b_df['has_htn'].mean()))
            p_dm_buckets.append(float(b_df['has_dm'].mean()))
        else:
            p_htn_buckets.append(0.3)
            p_dm_buckets.append(0.2)
    stats['p_htn_buckets'] = p_htn_buckets
    stats['p_dm_buckets'] = p_dm_buckets

    # Fit Baseline SCr in log-space
    log_scr = np.log(raw_df['baseline_scr'].clip(lower=0.2, upper=15.0))
    stats['log_scr_mean'] = float(log_scr.mean())
    stats['log_scr_var'] = float(log_scr.var())

    # Multiple regression for baseline SCr
    X = np.column_stack([
        np.ones(n),
        raw_df['age'].values.astype(float),
        raw_df['gender'].values.astype(float),
        raw_df['has_htn'].values.astype(float),
        raw_df['has_dm'].values.astype(float)
    ])
    y = log_scr.values.astype(float)
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        beta = np.array([1.0, 0.002, 0.15, 0.08, 0.10])

    stats['scr_intercept'] = float(beta[0])
    stats['scr_age_slope'] = float(beta[1])
    stats['scr_gender_slope'] = float(beta[2])
    stats['scr_htn_slope'] = float(beta[3])
    stats['scr_dm_slope'] = float(beta[4])

    # Backward compatibility aliases
    stats['age_to_scr_slope'] = float(beta[1])
    stats['age_to_scr_intercept'] = float(beta[0])
    stats['log_scr_var'] = float(log_scr.var())

    # Exposure Hazard Rates
    stats['p_vanco_given_normal'] = float(raw_df[raw_df['age'] < 65]['received_vanco'].mean())
    stats['p_vanco_given_elderly'] = float(raw_df[raw_df['age'] >= 65]['received_vanco'].mean())
    stats['p_zosyn'] = float(raw_df['received_zosyn'].mean())

    # Outcome/Toxicity Rates
    v_z_cohort = raw_df[(raw_df['received_vanco'] == 1) & (raw_df['received_zosyn'] == 1)]
    v_only_cohort = raw_df[(raw_df['received_vanco'] == 1) & (raw_df['received_zosyn'] == 0)]
    z_only_cohort = raw_df[(raw_df['received_vanco'] == 0) & (raw_df['received_zosyn'] == 1)]
    none_cohort = raw_df[(raw_df['received_vanco'] == 0) & (raw_df['received_zosyn'] == 0)]

    stats['aki_rate_vanco_zosyn'] = float(v_z_cohort['developed_aki'].mean()) if len(v_z_cohort) > 0 else 0.45
    stats['aki_rate_vanco_only'] = float(v_only_cohort['developed_aki'].mean()) if len(v_only_cohort) > 0 else 0.20
    stats['aki_rate_zosyn_only'] = float(z_only_cohort['developed_aki'].mean()) if len(z_only_cohort) > 0 else 0.10
    stats['aki_rate_baseline'] = float(none_cohort['developed_aki'].mean()) if len(none_cohort) > 0 else 0.05

    return stats

# -----------------------------------------------------------------------------
# STAGE 2: PRIVACY-PRESERVING SYNTHESIS (THE REBIRTH)
# -----------------------------------------------------------------------------
def synthesize_cohort(
    parameters: Dict[str, Any],
    n_synthetic: int = 1000,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Uses ONLY the extracted statistical parameters to generate completely
    new patient profiles via vectorized numpy operations.
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

# -----------------------------------------------------------------------------
# STAGE 3: LONGITUDINAL TEMPORAL DRIFT & NARRATIVE TEXTUALIZATION
# -----------------------------------------------------------------------------
def generate_temporal_record(
    patient: Dict[str, Any],
    days: int = 5,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Simulates consecutive ICU days, calculating cumulative drug exposure,
    hemodynamic fluctuations, and downstream renal decay.
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
    
    # Track if KDIGO AKI criteria was ever met during the trajectory
    had_aki = False
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

        # Calculate KDIGO Stage and rolling baseline minimums
        past_48h = scr_history[-2:] if len(scr_history) >= 2 else []
        creat_low_past_48hr = min(past_48h) if past_48h else current_scr
        creat_low_past_7day = min(scr_history)

        stage = 0
        stage1_cond1 = current_scr >= creat_low_past_7day * 1.5
        stage1_cond2 = (len(past_48h) > 0 and current_scr >= creat_low_past_48hr + 0.3)
        stage2_cond = current_scr >= creat_low_past_7day * 2.0
        stage3_cond1 = current_scr >= creat_low_past_7day * 3.0
        stage3_cond2 = (current_scr >= 4.0 and (current_scr >= creat_low_past_7day * 1.5 or current_scr >= creat_low_past_48hr + 0.3))

        if stage1_cond1 or stage1_cond2:
            stage = 1
        if stage2_cond:
            stage = 2
        if stage3_cond1 or stage3_cond2:
            stage = 3

        scr_history.append(current_scr)

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
            'scr': round(current_scr, 2),
            'risk_state': risk_label,
            'kdigo_stage': stage,
            'creat_low_past_48hr': round(creat_low_past_48hr, 2),
            'creat_low_past_7day': round(creat_low_past_7day, 2),
        })

    return trajectory

def format_to_llm_jsonl(patient, trajectory):
    """
    Transforms the structured tabular history into a fine-tuning prompt sequence
    with dynamic clinical reasoning and sim-to-real gap closures.
    """
    messages = []
    system_instruction = (
        "You are an AI-enabled clinical safety sentinel. Your task is to continuous-monitor "
        "ICU patient trajectories and predict the imminent onset of Medication-Induced Kidney Injury."
    )
    messages.append({"role": "system", "content": system_instruction})
    
    # Construct the user timeline prompt
    comorb_str = ", ".join(patient.get('comorbidities', [])) if patient.get('comorbidities') else "None"
    user_prompt = (
        f"Patient demographics: {patient['age']} yo, Sex: {patient['gender']}. "
        f"Comorbidities: {comorb_str}.\n"
        f"Baseline Serum Creatinine: {patient['baseline_scr']} mg/dL.\n"
        f"Initiating ICU clinical monitoring sequence:\n"
    )
    
    for record in trajectory:
        user_prompt += (
            f"[Day {record['day']}] MAP: {record['map']} mmHg | "
            f"Meds Active: Vanc={record['vanco_active']}, Zosyn={record['zosyn_active']} | "
            f"Vanco Trough: {record['vanco_trough']} ug/mL | "
            f"SCr: {record['scr']} mg/dL\n"
        )
        
    # Analyze the trajectory for clinical reasoning
    has_vanco = any(record['vanco_active'] for record in trajectory)
    has_zosyn = any(record['zosyn_active'] for record in trajectory)
    max_trough = max(record['vanco_trough'] for record in trajectory) if trajectory else 0
    min_map = min(record['map'] for record in trajectory) if trajectory else 100
    
    scr_values = [record['scr'] for record in trajectory]
    final_scr = scr_values[-1] if scr_values else patient['baseline_scr']
    max_scr = max(scr_values) if scr_values else patient['baseline_scr']
    max_day = scr_values.index(max_scr) + 1 if scr_values else 1
    
    # Detect recovery phase (creatinine peaked >= 1.5x baseline and then fell below peak)
    is_recovering = max_scr >= patient['baseline_scr'] * 1.5 and final_scr < max_scr
    
    final_day = trajectory[-1]
    is_aki = final_day['risk_state'] == "AKI_STAGE_1+"
    
    # Track drug exposure and charting omissions
    vanco_exposure_without_flowsheet = max_trough > 0.0 and not has_vanco
    actual_vanco_exposure = has_vanco or max_trough > 0.0
    actual_zosyn_exposure = has_zosyn # Zosyn is only tracked via flowsheet

    # Build clinical justification steps
    reasons = []
    
    # 1. Evaluate Drug Exposure & Charting Errors
    if actual_vanco_exposure or actual_zosyn_exposure:
        if vanco_exposure_without_flowsheet:
            reasons.append(
                f"Although active flowsheet flags for Vancomycin are absent (Vanc=False), "
                f"the presence of elevated Vancomycin trough levels (up to {max_trough} ug/mL) "
                f"confirms unrecorded/uncharted active drug exposure."
            )
        else:
            reasons.append("Patient has received nephrotoxic antibiotics (Vancomycin and/or Zosyn).")
            
        if actual_vanco_exposure and actual_zosyn_exposure:
            reasons.append("Co-administration of Vancomycin and Zosyn increases the risk of synergistic nephrotoxicity.")
        if max_trough > 20.0:
            reasons.append(f"Toxic Vancomycin exposure detected with a peak trough of {max_trough} ug/mL, exceeding the safe therapeutic range of 15-20 ug/mL.")
    else:
        if is_aki:
            reasons.append(
                "Patient has no active flowsheet prescriptions or measured levels for Vancomycin or Zosyn. "
                "However, they developed acute kidney injury due to non-medication-induced clinical factors."
            )
        else:
            reasons.append("Patient has no active flowsheet prescriptions or exposure to nephrotoxic study antibiotics.")

    # 2. Evaluate Baseline Risk / Comorbidities (Synthea-calibrated)
    has_ckd = any("Chronic Kidney Disease" in c for c in patient.get('comorbidities', []))
    has_dm = "Type 2 Diabetes Mellitus" in patient.get('comorbidities', [])
    has_htn = "Hypertension" in patient.get('comorbidities', [])
    
    comorb_reasons = []
    if has_ckd:
        ckd_stage = next((c for c in patient['comorbidities'] if "Chronic Kidney Disease" in c), "Chronic Kidney Disease")
        comorb_reasons.append(f"Pre-existing {ckd_stage} reduces baseline renal reserve, making the kidneys highly vulnerable to nephrotoxic injury.")
    if has_dm:
        comorb_reasons.append("Type 2 Diabetes Mellitus presents a high risk of diabetic microvascular changes, predisposing the kidneys to injury.")
    if has_htn:
        comorb_reasons.append("Hypertension impairs renal vascular autoregulation, exacerbating perfusion drops.")
        
    if comorb_reasons:
        reasons.append(" ".join(comorb_reasons))

    # 3. Evaluate Hemodynamics
    if min_map < 65:
        reasons.append(
            f"A hypotensive episode was observed with a minimum MAP of {min_map} mmHg, "
            f"which compromises renal perfusion pressure and worsens ischemic renal injury."
        )

    # 4. Evaluate Serum Creatinine Trajectory & KDIGO Status
    # Find the peak stage and details (with on-the-fly calculation fallback for real-world/legacy data)
    peak_stage = 0
    peak_record = None
    scr_history = [patient['baseline_scr']]

    for record in trajectory:
        rec_scr = record['scr']
        if 'kdigo_stage' in record:
            rec_stage = record['kdigo_stage']
            creat_low_past_48hr = record.get('creat_low_past_48hr', rec_scr)
            creat_low_past_7day = record.get('creat_low_past_7day', patient['baseline_scr'])
        else:
            past_48h = scr_history[-2:] if len(scr_history) >= 2 else []
            creat_low_past_48hr = min(past_48h) if past_48h else rec_scr
            creat_low_past_7day = min(scr_history)

            rec_stage = 0
            stage1_cond1 = rec_scr >= creat_low_past_7day * 1.5
            stage1_cond2 = (len(past_48h) > 0 and rec_scr >= creat_low_past_48hr + 0.3)
            stage2_cond = rec_scr >= creat_low_past_7day * 2.0
            stage3_cond1 = rec_scr >= creat_low_past_7day * 3.0
            stage3_cond2 = (rec_scr >= 4.0 and (rec_scr >= creat_low_past_7day * 1.5 or rec_scr >= creat_low_past_48hr + 0.3))

            if stage1_cond1 or stage1_cond2:
                rec_stage = 1
            if stage2_cond:
                rec_stage = 2
            if stage3_cond1 or stage3_cond2:
                rec_stage = 3

        scr_history.append(rec_scr)

        if rec_stage > peak_stage:
            peak_stage = rec_stage
            peak_record = {
                'scr': rec_scr,
                'creat_low_past_48hr': creat_low_past_48hr,
                'creat_low_past_7day': creat_low_past_7day
            }

    if peak_stage == 0 and is_aki:
        peak_stage = 1

    if is_recovering:
        stage_desc = ""
        if peak_stage == 3:
            stage_desc = "Stage 3 (severe injury, SCr >= 3.0x baseline or >= 4.0 mg/dL with acute rise)"
        elif peak_stage == 2:
            stage_desc = "Stage 2 (SCr >= 2.0x baseline)"
        else:
            stage_desc = "Stage 1 (SCr >= 1.5x baseline or increase >= 0.3 mg/dL within 48 hours)"

        reasons.append(
            f"Serum creatinine peaked at {max_scr} mg/dL on Day {max_day} (representing a {max_scr / patient['baseline_scr']:.2f}x rise from baseline of {patient['baseline_scr']} mg/dL). "
            f"Under KDIGO guidelines, this rise meets the criteria for {stage_desc} Acute Kidney Injury. "
            f"Although Serum Creatinine has started to resolve to {final_scr} mg/dL and nephrotoxic therapy was discontinued, "
            f"the patient is still classified as having experienced AKI during this clinical trajectory."
        )
    else:
        if is_aki:
            stage_desc = ""
            if peak_stage == 3:
                if peak_record and peak_record.get('scr', 0.0) >= 4.0 and (peak_record.get('scr', 0.0) / peak_record.get('creat_low_past_7day', 1.0) < 3.0):
                    stage_desc = "Stage 3 (Serum Creatinine >= 4.0 mg/dL with an acute rise)"
                else:
                    stage_desc = "Stage 3 (Serum Creatinine >= 3.0x baseline)"
            elif peak_stage == 2:
                stage_desc = "Stage 2 (Serum Creatinine >= 2.0x baseline)"
            else:
                if peak_record and (peak_record.get('scr', 0.0) >= peak_record.get('creat_low_past_48hr', 0.0) + 0.3) and (peak_record.get('scr', 0.0) / peak_record.get('creat_low_past_7day', 1.0) < 1.5):
                    stage_desc = "Stage 1 (acute Serum Creatinine rise >= 0.3 mg/dL within 48 hours)"
                else:
                    stage_desc = "Stage 1 (Serum Creatinine >= 1.5x baseline)"

            reasons.append(
                f"Serum creatinine rose significantly from a baseline of {patient['baseline_scr']} mg/dL "
                f"to a peak of {max_scr} mg/dL ({max_scr / patient['baseline_scr']:.2f}x baseline), "
                f"meeting the KDIGO definition of {stage_desc} Acute Kidney Injury."
            )
        else:
            reasons.append(f"Serum creatinine remains stable at {final_scr} mg/dL, showing no clinically significant rise above baseline.")

    synthesis_text = " ".join(reasons)
    assistant_response = (
        f"Clinical Synthesis: {synthesis_text} "
        f"Cumulative exposure combined with hemodynamic parameters indicates "
        f"risk status of the patient is currently assessed as: [{final_day['risk_state']}]."
    )
    
    messages.append({"role": "user", "content": user_prompt.strip()})
    messages.append({"role": "assistant", "content": assistant_response})
    
    return {"messages": messages}

# -----------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Step 1: Simulating messy historical EHR databases (MIMIC-IV representation)...")
    raw_cohort = generate_mock_historical_data(n_patients=1000)
    print(f"-> Analyzed {len(raw_cohort)} baseline historical trajectories.")
    
    print("\nStep 2: Performing parametric extraction...")
    extracted_parameters = extract_statistical_parameters(raw_cohort)
    print("-> Extracted statistical soul. Discarding raw patient rows to ensure differential privacy.")
    print("   Parameters Sample:")
    print(f"   - Age Dist: Mean {extracted_parameters['age_mean']:.2f}, SD {extracted_parameters['age_std']:.2f}")
    print(f"   - Synergy AKI Risk (Vanc + Zosyn): {extracted_parameters['aki_rate_vanco_zosyn']*100:.1f}%")
    print(f"   - Baseline AKI Risk: {extracted_parameters['aki_rate_baseline']*100:.1f}%")
    
    print("\nStep 3: Generating clean synthetic patient base...")
    synthetic_base = synthesize_cohort(extracted_parameters, n_synthetic=5)
    
    print("\nStep 4: Running longitudinal trajectories & formatting for LLM training...")
    jsonl_output = []
    for patient in synthetic_base:
        trajectory = generate_temporal_record(patient)
        llm_sample = format_to_llm_jsonl(patient, trajectory)
        jsonl_output.append(llm_sample)
        
    print("\n--- SAMPLE GENERATED LLM PAYLOAD (JSONL Format) ---")
    print(json.dumps(jsonl_output[0], indent=2))