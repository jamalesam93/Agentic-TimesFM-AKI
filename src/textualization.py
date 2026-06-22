import json
import pandas as pd
from typing import Dict, Any, List

def format_to_llm_jsonl(patient: pd.Series, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Transforms the structured tabular history into a fine-tuning prompt sequence.
    
    Args:
        patient (pd.Series): The baseline clinical data of the patient.
        trajectory (List[Dict[str, Any]]): The simulated temporal ICU trajectory.
        
    Returns:
        Dict[str, Any]: A dictionary structured as an LLM conversation format.
    """
    messages = []
    system_instruction = (
        "You are an AI-enabled clinical safety sentinel. Your task is to continuous-monitor "
        "ICU patient trajectories and predict the imminent onset of Medication-Induced Kidney Injury."
    )
    messages.append({"role": "system", "content": system_instruction})
    
    # Construct the user timeline prompt
    user_prompt = (
        f"Patient demographics: {patient['age']} yo, Sex: {patient['gender']}. "
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
        
    # The expected output showing clinical reasoning and the target class
    final_day = trajectory[-1]
    assistant_response = (
        f"Clinical Synthesis: Patient has received nephrotoxic antibiotics. "
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
