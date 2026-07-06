import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import data_extraction_real as ehr
from src import generator

def main():
    print("Loading real raw data for varying epsilon analysis...")
    # Load 1000 real patients
    raw_df = ehr.load_real_historical_data(n_patients=1000, seed=42)
    
    # Create feature representations for evaluation (the "true" data)
    raw_df['is_male'] = raw_df['gender']
    raw_df['has_ckd'] = raw_df['baseline_scr'] >= 1.5
    features = ['age', 'is_male', 'baseline_scr', 'has_htn', 'has_dm', 'has_ckd', 'received_vanco', 'received_zosyn']
    
    # Split the real data into train/test (we will evaluate the synthetic models on the real test set)
    X = raw_df[features]
    y = raw_df['developed_aki']
    _, X_test_real, _, y_test_real = train_test_split(X, y, test_size=0.3, random_state=123)
    
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    aucs = []
    
    # Also evaluate a model trained on the raw (non-private) data as an upper bound
    X_train_raw, _, y_train_raw, _ = train_test_split(X, y, test_size=0.3, random_state=123)
    rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_baseline.fit(X_train_raw, y_train_raw)
    auc_baseline = roc_auc_score(y_test_real, rf_baseline.predict_proba(X_test_real)[:, 1])
    print(f"Non-private baseline AUC: {auc_baseline:.3f}")

    for eps in epsilons:
        print(f"\n--- Testing epsilon = {eps} ---")
        try:
            stats = ehr.extract_statistical_parameters(raw_df, epsilon=eps)
            synthetic_patients = generator.synthesize_cohort(stats, n_synthetic=3000, seed=123)
            df_synth = pd.DataFrame(synthetic_patients)
            
            df_synth['is_male'] = (df_synth['gender'] == 'M').astype(int)
            df_synth['has_htn'] = df_synth['comorbidities'].apply(lambda x: 1 if 'Hypertension' in x else 0)
            df_synth['has_dm'] = df_synth['comorbidities'].apply(lambda x: 1 if 'Type 2 Diabetes Mellitus' in x else 0)
            df_synth['has_ckd'] = df_synth['comorbidities'].apply(lambda x: 1 if any('Chronic Kidney Disease' in c for c in x) else 0)
            
            X_train_synth = df_synth[features]
            y_train_synth = df_synth['developed_aki']
            
            rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            rf.fit(X_train_synth, y_train_synth)
            
            auc = roc_auc_score(y_test_real, rf.predict_proba(X_test_real)[:, 1])
            print(f"AUC at eps={eps}: {auc:.3f}")
            aucs.append(auc)
        except Exception as e:
            print(f"Failed at eps={eps}: {e}")
            aucs.append(np.nan)
            
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, aucs, marker='o', linewidth=2, color='blue', label='DP Synthetic Model')
    plt.axhline(y=auc_baseline, color='red', linestyle='--', label=f'Non-private Model (AUC={auc_baseline:.3f})')
    
    plt.xscale('log')
    plt.xlabel('Privacy Budget (ε) - Log Scale')
    plt.ylabel('ROC AUC on Real Holdout Data')
    plt.title('Privacy-Utility Tradeoff\nImpact of Differential Privacy on Model Performance')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    plots_dir = os.path.join(root_dir, "plots", "real_world")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "epsilon_tradeoff.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved tradeoff curve to {out_path}")

if __name__ == "__main__":
    main()
