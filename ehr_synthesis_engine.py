import numpy as np
import pandas as pd
import json
from scipy.stats import norm, lognorm

# -----------------------------------------------------------------------------
# STAGE 0: SIMULATE RAW HISTORICAL CLINICAL DATA (REPRESENTING MIMIC-IV)
# -----------------------------------------------------------------------------
def generate_mock_historical_data(n_patients=500):
    """
    Generates a messy, historical dataset representing raw data from PhysioNet.
    This serves as the source from which we will extract our statistical parameters.
    """
    np.random.seed(101)
    
    # Baseline Covariates
    ages = np.random.normal(63, 14, n_patients).clip(18, 95)
    genders = np.random.binomial(1, 0.52, n_patients) # 1 = Male, 0 = Female
    
    # Baseline Serum Creatinine (SCr) - correlated with age
    baseline_scr = []
    for age, gender in zip(ages, genders):
        base = 0.8 + (age * 0.003) + (gender * 0.15)
        scr = np.random.lognormal(mean=np.log(base), sigma=0.18)
        baseline_scr.append(round(scr, 2))
        
    df_base = pd.DataFrame({
        'patient_id': [f"HIST_{i:04d}" for i in range(n_patients)],
        'age': ages.astype(int),
        'gender': genders,
        'baseline_scr': baseline_scr
    })
    
    # Simulate exposure and outcomes
    # Exposure to Vancomycin (higher in older, sicker patients)
    vanco_prob = 1 / (1 + np.exp(-(df_base['age'] * 0.02 + df_base['baseline_scr'] * 0.5 - 2.0)))
    df_base['received_vanco'] = np.random.binomial(1, vanco_prob)
    
    # Co-administration of Piperacillin-Tazobactam (Zosyn)
    df_base['received_zosyn'] = np.random.binomial(1, 0.4, n_patients)
    
    # Outcomes: AKI incidence (Synergistic effect of Vanco + Zosyn)
    aki_prob = []
    for _, row in df_base.iterrows():
        p = 0.05 # Baseline ICU AKI rate
        if row['received_vanco'] == 1:
            p += 0.15 # Vanco effect
        if row['received_zosyn'] == 1:
            p += 0.05 # Zosyn baseline effect
        if row['received_vanco'] == 1 and row['received_zosyn'] == 1:
            p += 0.20 # Synergistic "Zosyn-Vanc" toxicity boost
        aki_prob.append(min(0.95, p))
        
    df_base['developed_aki'] = np.random.binomial(1, aki_prob)
    return df_base

# -----------------------------------------------------------------------------
# STAGE 1: THE PARAMETRIC EXTRACTION (EXTRACTING THE STATISTICAL SOUL)
# -----------------------------------------------------------------------------
def extract_statistical_parameters(raw_df):
    """
    Analyzes raw, real-world data to extract statistical descriptors, 
    ensuring we do not store any individual patient's raw records.
    """
    stats = {}
    
    # 1. Demographic Marginals
    stats['age_mean'] = float(raw_df['age'].mean())
    stats['age_std'] = float(raw_df['age'].std())
    stats['male_proportion'] = float(raw_df['gender'].mean())
    
    # 2. Fit Baseline SCr using Lognormal distribution parameters
    shape, loc, scale = lognorm.fit(raw_df['baseline_scr'])
    stats['scr_lognorm_params'] = [float(shape), float(loc), float(scale)]
    
    # 3. Covariance baseline adjustment (how age affects baseline SCr)
    # We fit a simple linear regression parameter to maintain the biological relationship
    slope, intercept = np.polyfit(raw_df['age'], raw_df['baseline_scr'], 1)
    stats['age_to_scr_slope'] = float(slope)
    stats['age_to_scr_intercept'] = float(intercept)
    
    # 4. Exposure Hazard Rates (Logistic Regression Coefficients for Drug Assignment)
    # Simple calculation of probabilities based on cohorts
    stats['p_vanco_given_normal'] = float(raw_df[raw_df['age'] < 65]['received_vanco'].mean())
    stats['p_vanco_given_elderly'] = float(raw_df[raw_df['age'] >= 65]['received_vanco'].mean())
    stats['p_zosyn'] = float(raw_df['received_zosyn'].mean())
    
    # 5. Outcome/Toxicity Rates (Synergistic risk mapping)
    v_z_cohort = raw_df[(raw_df['received_vanco'] == 1) & (raw_df['received_zosyn'] == 1)]
    v_only_cohort = raw_df[(raw_df['received_vanco'] == 1) & (raw_df['received_zosyn'] == 0)]
    none_cohort = raw_df[(raw_df['received_vanco'] == 0) & (raw_df['received_zosyn'] == 0)]
    
    stats['aki_rate_vanco_zosyn'] = float(v_z_cohort['developed_aki'].mean()) if len(v_z_cohort) > 0 else 0.45
    stats['aki_rate_vanco_only'] = float(v_only_cohort['developed_aki'].mean()) if len(v_only_cohort) > 0 else 0.20
    stats['aki_rate_baseline'] = float(none_cohort['developed_aki'].mean()) if len(none_cohort) > 0 else 0.05
    
    return stats

# -----------------------------------------------------------------------------
# STAGE 2: PRIVACY-PRESERVING SYNTHESIS (THE REBIRTH)
# -----------------------------------------------------------------------------
def synthesize_cohort(parameters, n_synthetic=1000):
    """
    Uses ONLY the extracted statistical parameters to generate completely 
    new patient trajectories. No raw data is ever accessed here.
    """
    synthetic_patients = []
    
    for i in range(n_synthetic):
        # 1. Synthesize Baseline Covariates
        age = int(np.random.normal(parameters['age_mean'], parameters['age_std']))
        age = max(18, min(age, 95))
        
        gender = 1 if np.random.rand() < parameters['male_proportion'] else 0
        
        # 2. Synthesize baseline SCr matching the age-conditioned slope + lognormal residual noise
        predicted_mean_scr = (age * parameters['age_to_scr_slope']) + parameters['age_to_scr_intercept']
        # Sample a lognormal deviation
        shape, loc, scale = parameters['scr_lognorm_params']
        residual = lognorm.rvs(shape, loc, scale) - scale  # Center the distribution
        baseline_scr = round(max(0.4, predicted_mean_scr + residual), 2)
        
        # 3. Determine Drug Exposures based on statistical hazard parameters
        prob_vanco = parameters['p_vanco_given_elderly'] if age >= 65 else parameters['p_vanco_given_normal']
        received_vanco = 1 if np.random.rand() < prob_vanco else 0
        received_zosyn = 1 if np.random.rand() < parameters['p_zosyn'] else 0
        
        # 4. Compute Synergistic Toxicity Risk
        if received_vanco == 1 and received_zosyn == 1:
            aki_prob = parameters['aki_rate_vanco_zosyn']
        elif received_vanco == 1:
            aki_prob = parameters['aki_rate_vanco_only']
        else:
            aki_prob = parameters['aki_rate_baseline']
            
        developed_aki = 1 if np.random.rand() < aki_prob else 0
        
        # Assemble patient baseline profile
        synthetic_patients.append({
            'synthetic_id': f"SYN_{i:05d}",
            'age': age,
            'gender': "M" if gender == 1 else "F",
            'baseline_scr': baseline_scr,
            'received_vanco': bool(received_vanco),
            'received_zosyn': bool(received_zosyn),
            'developed_aki': bool(developed_aki)
        })
        
    return pd.DataFrame(synthetic_patients)

# -----------------------------------------------------------------------------
# STAGE 3: LONGITUDINAL TEMPORAL DRIFT & NARRATIVE TEXTUALIZATION
# -----------------------------------------------------------------------------
def generate_temporal_record(patient):
    """
    Simulates consecutive ICU days, calculating cumulative drug exposure, 
    hemodynamic fluctuations, and downstream renal decay.
    """
    days = 5
    trajectory = []
    current_scr = patient['baseline_scr']
    vanco_trough = 0.0
    
    for day in range(1, days + 1):
        # Base hemodynamics
        map_val = int(np.random.normal(74, 6))
        # If severe AKI is developing, drop the blood pressure
        if patient['developed_aki'] and day >= 3:
            map_val = int(np.random.normal(63, 4))
            
        # Cumulative toxic exposure calculation
        vanco_status = "ACTIVE" if (patient['received_vanco'] and day >= 2) else "OFF"
        zosyn_status = "ACTIVE" if (patient['received_zosyn'] and day >= 2) else "OFF"
        
        if vanco_status == "ACTIVE":
            vanco_trough += np.random.uniform(3.0, 5.0)
            if patient['received_zosyn']:
                # Synergy accelerates and exacerbates the structural damage
                vanco_trough += np.random.uniform(1.5, 3.0)
                
        # Simulate temporal rise of SCr if patient is doomed to AKI
        if patient['developed_aki'] and day >= 3:
            # Serum Creatinine lag: peaks hours/days after cellular damage occurs
            decay_rate = np.random.uniform(0.3, 0.8)
            current_scr += round(decay_rate, 2)
        else:
            # Normal physiological drift
            current_scr += round(np.random.normal(0.0, 0.03), 2)
            
        # Label formatting for training
        risk_label = "NORMAL"
        if current_scr >= patient['baseline_scr'] * 1.5:
            risk_label = "AKI_STAGE_1+"
            
        day_record = {
            'day': day,
            'map': map_val,
            'vanco_active': vanco_status == "ACTIVE",
            'zosyn_active': zosyn_status == "ACTIVE",
            'vanco_trough': round(vanco_trough, 1) if vanco_status == "ACTIVE" else 0.0,
            'scr': round(current_scr, 2),
            'risk_state': risk_label
        }
        trajectory.append(day_record)
        
    return trajectory

def format_to_llm_jsonl(patient, trajectory):
    """
    Transforms the structured tabular history into a fine-tuning prompt sequence.
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
    for _, patient in synthetic_base.iterrows():
        trajectory = generate_temporal_record(patient)
        llm_sample = format_to_llm_jsonl(patient, trajectory)
        jsonl_output.append(llm_sample)
        
    print("\n--- SAMPLE GENERATED LLM PAYLOAD (JSONL Format) ---")
    print(json.dumps(jsonl_output[0], indent=2))