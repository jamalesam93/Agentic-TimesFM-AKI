import pandas as pd
import json
import random
from pathlib import Path

# Paths
DATA_DIR = Path("data/real_world/eicu-collaborative-research-database-demo-2.0.1")
OUT_FILE = Path("data/eicu_eval_holdout.jsonl")

def main():
    print("Loading eICU tables...")
    # Load patient demographics
    df_pat = pd.read_csv(DATA_DIR / "patient.csv")
    
    # Load labs
    df_lab = pd.read_csv(DATA_DIR / "lab.csv")
    df_scr = df_lab[df_lab['labname'].str.contains('creatinine', case=False, na=False)]
    df_vtrough = df_lab[df_lab['labname'].str.contains('Vancomycin - trough', case=False, na=False)]
    
    # Load meds
    df_med = pd.read_csv(DATA_DIR / "medication.csv", low_memory=False)
    # Fill NA offsets with 0 or drop them
    df_med = df_med.dropna(subset=['drugstartoffset'])
    
    df_vanco_meds = df_med[df_med['drugname'].str.contains('vanco', case=False, na=False)]
    df_zosyn_meds = df_med[df_med['drugname'].str.contains('piperacillin|zosyn', case=False, na=False)]
    
    # Load vitals
    df_vit = pd.read_csv(DATA_DIR / "vitalAperiodic.csv")
    df_map = df_vit.dropna(subset=['noninvasivemean'])

    # Find valid stays (must have at least 1 SCr)
    valid_stays = df_scr['patientunitstayid'].unique()
    random.seed(42)
    sample_stays = random.sample(list(valid_stays), min(300, len(valid_stays)))
    
    dataset = []
    
    print(f"Processing {len(sample_stays)} sampled stays...")
    for stay_id in sample_stays:
        # Demographics
        pat_info = df_pat[df_pat['patientunitstayid'] == stay_id]
        if pat_info.empty:
            continue
        age = str(pat_info['age'].values[0]).replace('> 89', '90')
        gender = pat_info['gender'].values[0]
        if gender not in ['Male', 'Female']:
            gender = 'Unknown'
        sex_str = 'M' if gender == 'Male' else 'F' if gender == 'Female' else 'U'
        
        # SCr
        stay_scr = df_scr[df_scr['patientunitstayid'] == stay_id].sort_values('labresultoffset')
        if len(stay_scr) < 2:
            continue # Need at least 2 to have a trajectory
            
        baseline_scr = stay_scr.iloc[0]['labresult']
        if pd.isna(baseline_scr) or baseline_scr <= 0:
            continue
            
        # Group by day
        max_days = min(5, int((stay_scr['labresultoffset'].max() // 1440) + 1))
        if max_days < 2:
            continue
            
        stay_vtrough = df_vtrough[df_vtrough['patientunitstayid'] == stay_id]
        stay_vanco_meds = df_vanco_meds[df_vanco_meds['patientunitstayid'] == stay_id]
        stay_zosyn_meds = df_zosyn_meds[df_zosyn_meds['patientunitstayid'] == stay_id]
        stay_map = df_map[df_map['patientunitstayid'] == stay_id]
        
        prompt_lines = [
            f"Patient demographics: {age} yo, Sex: {sex_str}. Baseline Serum Creatinine: {baseline_scr:.1f} mg/dL.",
            "Initiating ICU clinical monitoring sequence:"
        ]
        
        received_vanco = False
        received_zosyn = False
        peak_scr = baseline_scr
        final_scr = baseline_scr
        
        for day in range(max_days):
            start_min = day * 1440
            end_min = (day + 1) * 1440
            
            # Day SCr
            day_scr_vals = stay_scr[(stay_scr['labresultoffset'] >= start_min) & (stay_scr['labresultoffset'] < end_min)]
            day_scr = day_scr_vals['labresult'].max() if not day_scr_vals.empty else final_scr
            final_scr = day_scr
            peak_scr = max(peak_scr, day_scr)
            
            # Day MAP
            day_map_vals = stay_map[(stay_map['observationoffset'] >= start_min) & (stay_map['observationoffset'] < end_min)]
            day_map = int(day_map_vals['noninvasivemean'].mean()) if not day_map_vals.empty else 72
            
            # Day Vanco Trough
            day_vt_vals = stay_vtrough[(stay_vtrough['labresultoffset'] >= start_min) & (stay_vtrough['labresultoffset'] < end_min)]
            day_vt = day_vt_vals['labresult'].max() if not day_vt_vals.empty else 0.0
            
            # Active meds
            vanc_active = not stay_vanco_meds[(stay_vanco_meds['drugstartoffset'] <= end_min) & ((stay_vanco_meds['drugstopoffset'] >= start_min) | stay_vanco_meds['drugstopoffset'].isna())].empty
            zosyn_active = not stay_zosyn_meds[(stay_zosyn_meds['drugstartoffset'] <= end_min) & ((stay_zosyn_meds['drugstopoffset'] >= start_min) | stay_zosyn_meds['drugstopoffset'].isna())].empty
            
            if vanc_active: received_vanco = True
            if zosyn_active: received_zosyn = True
            
            prompt_lines.append(f"[Day {day+1}] MAP: {day_map} mmHg | Meds Active: Vanc={vanc_active}, Zosyn={zosyn_active} | Vanco Trough: {day_vt:.1f} ug/mL | SCr: {day_scr:.1f} mg/dL")
            
        # Determine ground truth KDIGO
        # KDIGO Stage 1: SCr >= 1.5 * baseline OR increase of >= 0.3
        is_aki = (peak_scr >= 1.5 * baseline_scr) or (peak_scr >= baseline_scr + 0.3)
        gt_label = "AKI_STAGE_1+" if is_aki else "NORMAL"
        
        user_content = "\n".join(prompt_lines)
        system_content = "You are an AI-enabled clinical safety sentinel. Your task is to continuous-monitor ICU patient trajectories and predict the imminent onset of Medication-Induced Kidney Injury."
        assistant_content = f"Clinical Synthesis: Patient has received nephrotoxic antibiotics. Cumulative exposure combined with hemodynamic parameters indicates risk status of the patient is currently assessed as: [{gt_label}]."
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        meta = {
            "source": "eICU Demo 2.0.1",
            "stay_id": int(stay_id),
            "baseline_scr": float(baseline_scr),
            "final_scr": float(final_scr),
            "final_risk": gt_label,
            "n_trajectory_days": int(max_days),
            "received_vanco": bool(received_vanco),
            "received_zosyn": bool(received_zosyn)
        }
        
        dataset.append({
            "messages": messages,
            "_meta": meta
        })

    # Save to JSONL
    print(f"Generated {len(dataset)} trajectories.")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
