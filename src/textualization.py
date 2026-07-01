import json
import pandas as pd
from typing import Dict, Any, List

def format_to_llm_jsonl(patient: pd.Series, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Transforms the structured tabular history into a fine-tuning prompt sequence
    with dynamic clinical reasoning and sim-to-real gap closures.
    
    Args:
        patient (pd.Series): The baseline clinical data of the patient.
        trajectory (List[Dict[str, Any]]): The simulated temporal ICU trajectory.
        
    Returns:
        Dict[str, Any]: A dictionary structured as an LLM conversation format.
    """
    messages = []
    system_instruction = (
        "You are an AI-enabled clinical safety sentinel. Your task is to continuous-monitor "
        "ICU patient trajectories and predict the imminent onset of Medication-Induced Kidney Injury.\n\n"
    )
    
    # Construct the user timeline prompt
    comorb_str = ", ".join(patient.get('comorbidities', [])) if patient.get('comorbidities') else "None"
    user_prompt = (
        f"{system_instruction}"
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


def format_to_clinical_markdown(patient: pd.Series, trajectory: List[Dict[str, Any]]) -> str:
    """
    Converts a patient profile and temporal trajectory into a clinical monitoring report
    in Markdown format.
    
    Args:
        patient (pd.Series): The baseline clinical data of the patient.
        trajectory (List[Dict[str, Any]]): The simulated temporal ICU trajectory.
        
    Returns:
        str: Markdown formatted patient report.
    """
    gender_full = "Male" if patient['gender'] == "M" else "Female"
    aki_status = "DEVELOPED ACI (Acute Kidney Injury)" if patient['developed_aki'] else "NORMAL RENAL FUNCTION PRESERVED"
    final_day = trajectory[-1]
    
    md = []
    md.append(f"# ICU CLINICAL MONITORING CHART - PATIENT {patient['synthetic_id']}")
    md.append("=" * 60)
    md.append("")
    md.append("## 1. Demographics & Baseline Clinical Data")
    md.append(f"- **Patient ID**: {patient['synthetic_id']}")
    md.append(f"- **Age**: {patient['age']} years")
    md.append(f"- **Gender**: {gender_full}")
    md.append(f"- **Baseline Serum Creatinine (SCr)**: {patient['baseline_scr']} mg/dL")
    md.append("")
    md.append("## 2. Longitudinal Temporal ICU Flowsheet")
    md.append("| Day | MAP (mmHg) | Vanco Active | Zosyn Active | Vanco Trough (ug/mL) | SCr (mg/dL) | Risk State |")
    md.append("|:---:|:----------:|:------------:|:------------:|:--------------------:|:----------:|:----------:|")
    
    for day in trajectory:
        md.append(
            f"| Day {day['day']} | {day['map']} | {day['vanco_active']} | {day['zosyn_active']} | "
            f"{day['vanco_trough']:.1f} | {day['scr']:.2f} | **{day['risk_state']}** |"
        )
        
    md.append("")
    md.append("## 3. Clinical Synthesis & Risk Profile")
    md.append(f"- **Vancomycin Exposure**: {'Yes' if patient['received_vanco'] else 'No'}")
    md.append(f"- **Zosyn (Pip/Tazo) Exposure**: {'Yes' if patient['received_zosyn'] else 'No'}")
    
    # Explain clinical synergy context
    if patient['received_vanco'] and patient['received_zosyn']:
        md.append("- **Nephrotoxic Risk Assessment**: HIGH RISK. Patient was prescribed a combination of **Vancomycin + Zosyn (Piperacillin-Tazobactam)**. Clinically, this combination has a known synergistic drug-drug interaction which significantly escalates the risk of acute kidney injury (AKI) compared to either drug alone.")
    elif patient['received_vanco']:
        md.append("- **Nephrotoxic Risk Assessment**: MODERATE RISK. Patient received Vancomycin monotherapy. Close therapeutic drug monitoring (TDM) is indicated to maintain troughs between 15-20 ug/mL.")
    elif patient['received_zosyn']:
        md.append("- **Nephrotoxic Risk Assessment**: LOW-TO-MODERATE RISK. Patient received Zosyn monotherapy. SCr should be monitored periodically.")
    else:
        md.append("- **Nephrotoxic Risk Assessment**: LOW RISK. No exposure to nephrotoxic study drugs.")
        
    md.append(f"- **AKI Outcome Status**: `{aki_status}`")
    md.append(f"- **Final Serum Creatinine**: {final_day['scr']:.2f} mg/dL (Baseline: {patient['baseline_scr']:.2f} mg/dL)")
    
    # Calculate KDIGO-based creatinine ratio
    scr_ratio = final_day['scr'] / patient['baseline_scr']
    md.append(f"- **Max SCr fold-increase**: {scr_ratio:.2f}x baseline")
    
    md.append("")
    md.append("-" * 60)
    md.append("*Generated by the Automated EHR Synthesis Engine Sentinel Module.*")
    
    return "\n".join(md)

def format_to_timesfm_dataset(patient: Dict[str, Any], trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Transforms the clinical record into a flat dictionary suitable for converting
    into a pandas DataFrame or Arrow table for TimesFM time-series forecasting.
    
    Args:
        patient (Dict[str, Any]): The baseline clinical data of the patient.
        trajectory (List[Dict[str, Any]]): The simulated temporal ICU trajectory.
        
    Returns:
        Dict[str, Any]: A flattened dictionary representing the patient's timeseries.
    """
    # Extract timeseries variables
    map_series = [record['map'] for record in trajectory]
    vanco_trough_series = [record['vanco_trough'] for record in trajectory]
    scr_series = [record['scr'] for record in trajectory]
    
    # Extract dynamic covariates
    vanco_active_series = [1 if record['vanco_active'] else 0 for record in trajectory]
    zosyn_active_series = [1 if record['zosyn_active'] else 0 for record in trajectory]
    
    return {
        'synthetic_id': patient['synthetic_id'],
        'age': patient['age'],
        'gender_encoded': 1 if patient['gender'] == 'M' else 0,
        'received_vanco': 1 if patient['received_vanco'] else 0,
        'received_zosyn': 1 if patient['received_zosyn'] else 0,
        'map': map_series,
        'vanco_trough': vanco_trough_series,
        'scr': scr_series,
        'vanco_active': vanco_active_series,
        'zosyn_active': zosyn_active_series
    }
