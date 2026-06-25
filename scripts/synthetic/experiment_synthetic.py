import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import data_extraction as ehr
from src import generator

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, "data", "synthetic")
    csv_clean_path = os.path.join(data_dir, "phd_proposal_synthetic_cohort.csv")
    csv_messy_path = os.path.join(data_dir, "phd_proposal_synthetic_cohort_messy.csv")
    
    if os.path.exists(csv_clean_path) and os.path.exists(csv_messy_path):
        print("-> Loading pre-synthesized audited cohort datasets from disk...")
        df_messy = pd.read_csv(csv_messy_path)
        
        # In the hybrid approach, we reconstruct the true clean baseline cohort from the messy cohort
        print("-> Reconstructing the true clean baseline cohort from messy data...")
        df = df_messy.copy()
        
        # 1. Restore age outliers
        outlier_ages = df['age'] > 100
        for idx in df[outlier_ages].index:
            age_val = df.loc[idx, 'age']
            if age_val % 10 == 0 and age_val > 100:
                df.loc[idx, 'age'] = int(age_val // 10)
            else:
                df.loc[idx, 'age'] = int(age_val - 100)
                
        # 2. Restore baseline_scr outliers and missing values
        clean_scr = df_messy[(df_messy['baseline_scr'].notna()) & (df_messy['baseline_scr'] <= 20.0) & (df_messy['baseline_scr'] > 0.0)]['baseline_scr']
        log_clean_scr = np.log(clean_scr)
        mean_log = log_clean_scr.mean()
        std_log = log_clean_scr.std()
        
        scr_outlier_or_missing = df['baseline_scr'].isna() | (df['baseline_scr'] > 20.0) | (df['baseline_scr'] == 0.0)
        np.random.seed(42)
        sampled_scr = np.random.lognormal(mean=mean_log, sigma=std_log, size=scr_outlier_or_missing.sum())
        df.loc[scr_outlier_or_missing, 'baseline_scr'] = np.round(sampled_scr, 2)
        
        # 3. Restore missing HTN and DM flags
        import ast
        def check_in_comorb(comorb_str, term):
            try:
                c_list = ast.literal_eval(comorb_str) if isinstance(comorb_str, str) else comorb_str
                return 1 if term in c_list else 0
            except Exception:
                return 0
                
        # First check if the comorbidity was already parsed
        htn_missing = df['has_htn'].isna()
        dm_missing = df['has_dm'].isna()
        
        df.loc[htn_missing, 'has_htn'] = df.loc[htn_missing, 'comorbidities'].apply(lambda x: check_in_comorb(x, 'Hypertension'))
        df.loc[dm_missing, 'has_dm'] = df.loc[dm_missing, 'comorbidities'].apply(lambda x: check_in_comorb(x, 'Type 2 Diabetes Mellitus'))
        
        # Fill remaining missing HTN and DM by overall prevalence sampling
        p_htn = df_messy['has_htn'].dropna().mean()
        p_dm = df_messy['has_dm'].dropna().mean()
        
        htn_still_missing = df['has_htn'].isna()
        dm_still_missing = df['has_dm'].isna()
        
        if htn_still_missing.sum() > 0:
            df.loc[htn_still_missing, 'has_htn'] = np.random.binomial(1, p_htn, size=htn_still_missing.sum())
        if dm_still_missing.sum() > 0:
            df.loc[dm_still_missing, 'has_dm'] = np.random.binomial(1, p_dm, size=dm_still_missing.sum())
            
        df['is_male'] = (df['gender'] == 'M').astype(int)
        df['has_ckd'] = df['comorbidities'].apply(lambda x: 1 if 'Chronic Kidney Disease' in str(x) else 0)
    else:
        print("1. Extracting parameters from generic mock EHR data...")
        raw_df = ehr.generate_mock_historical_data(n_patients=1000, seed=42)
        stats = ehr.extract_statistical_parameters(raw_df)
        
        print("2. Synthesizing 5,000 privacy-preserving patients...")
        synthetic_patients = generator.synthesize_cohort(stats, n_synthetic=5000, seed=123)
        
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(synthetic_patients)
        
        # Feature Engineering for ML
        df['is_male'] = (df['gender'] == 'M').astype(int)
        df['has_htn'] = df['comorbidities'].apply(lambda x: 1 if 'Hypertension' in x else 0)
        df['has_dm'] = df['comorbidities'].apply(lambda x: 1 if 'Type 2 Diabetes Mellitus' in x else 0)
        df['has_ckd'] = df['comorbidities'].apply(lambda x: 1 if any('Chronic Kidney Disease' in c for c in x) else 0)
        
        print("2b. Injecting realistic EHR noise (Missingness & Outliers)...")
        np.random.seed(999)
        df_messy = df.copy()
        n = len(df_messy)
        
        # Decoupled outlier masks
        age_outlier_mask = np.random.rand(n) < 0.005  # 0.5%
        scr_outlier_mask = np.random.rand(n) < 0.005  # 0.5% independent
        
        # Diversified age typos
        age_typo_types = np.random.rand(n)
        df_messy.loc[age_outlier_mask & (age_typo_types < 0.5), 'age'] *= 10
        df_messy.loc[age_outlier_mask & (age_typo_types >= 0.5), 'age'] += 100
        
        # Diversified SCr errors
        scr_errors = np.random.choice([99.9, 999.0, 0.0, 25.0], size=scr_outlier_mask.sum())
        df_messy.loc[scr_outlier_mask, 'baseline_scr'] = scr_errors
        
        # Missingness (MCAR)
        missing_scr_mask = np.random.rand(n) < 0.15
        df_messy.loc[missing_scr_mask, 'baseline_scr'] = np.nan
        
        # Coherent missingness for HTN and DM
        missing_htn_mask = np.random.rand(n) < 0.08
        df_messy.loc[missing_htn_mask, 'has_htn'] = np.nan
        def remove_htn(c_list): return [c for c in c_list if c != 'Hypertension']
        df_messy.loc[missing_htn_mask, 'comorbidities'] = df_messy.loc[missing_htn_mask, 'comorbidities'].apply(remove_htn)
        
        missing_dm_mask = np.random.rand(n) < 0.05
        df_messy.loc[missing_dm_mask, 'has_dm'] = np.nan
        def remove_dm(c_list): return [c for c in c_list if c != 'Type 2 Diabetes Mellitus']
        df_messy.loc[missing_dm_mask, 'comorbidities'] = df_messy.loc[missing_dm_mask, 'comorbidities'].apply(remove_dm)
        
        # Save the 'messy' dataset for the third-party agent and proposal appendix
        os.makedirs(data_dir, exist_ok=True)
        df_messy.to_csv(csv_messy_path, index=False)
        print(f"-> Saved messy dataset to {csv_messy_path}")

    print("2c. Performing Data Cleaning & Imputation...")
    def preprocess_ehr(input_df):
        """
        Preprocesses raw EHR data for ML modeling:
        1. Caps implausible ages at 100.
        2. Replaces laboratory sentinel values (e.g. 0.0, >20) with NaN.
        3. Median-imputes missing continuous labs.
        4. Assumes missing structured comorbidities (HTN, DM) are absent (MNAR logic).
        """
        out_df = input_df.copy()
        out_df.loc[out_df['age'] > 100, 'age'] = 100
        out_df.loc[out_df['baseline_scr'] > 20.0, 'baseline_scr'] = np.nan
        out_df.loc[out_df['baseline_scr'] == 0.0, 'baseline_scr'] = np.nan
        
        out_df['baseline_scr'] = out_df['baseline_scr'].fillna(out_df['baseline_scr'].median())
        out_df['has_htn'] = out_df['has_htn'].fillna(0)
        out_df['has_dm'] = out_df['has_dm'].fillna(0)
        return out_df

    df_clean_baseline = preprocess_ehr(df)  # The reconstructed clean cohort
    df_imputed = preprocess_ehr(df_messy)   # The messy cohort after cleaning

    features = ['age', 'is_male', 'baseline_scr', 'has_htn', 'has_dm', 'has_ckd', 'received_vanco', 'received_zosyn']
    
    print("3. Training Machine Learning Models for Robustness Comparison...")
    
    def train_and_eval(data, label=""):
        X = data[features]
        y = data['developed_aki']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"-> {label} ROC AUC Score: {auc_score:.3f}")
        return rf, X_test, y_test, y_pred_proba, auc_score
        
    rf_clean, X_test_c, y_test_c, prob_c, auc_c = train_and_eval(df_clean_baseline, "Clean Synthetic")
    rf_messy, X_test_m, y_test_m, prob_m, auc_m = train_and_eval(df_imputed, "Messy -> Imputed")
    
    print(f"-> Robustness Delta AUC: {auc_m - auc_c:+.3f}")
    
    # Plotting
    plots_dir = os.path.join(root_dir, "plots", "synthetic")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 4a. ROC Curve (Comparing Clean vs Imputed)
    fpr_c, tpr_c, _ = roc_curve(y_test_c, prob_c)
    fpr_m, tpr_m, _ = roc_curve(y_test_m, prob_m)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_c, tpr_c, color='darkorange', lw=2, label=f'Clean Baseline (AUC = {auc_c:.3f})')
    plt.plot(fpr_m, tpr_m, color='green', lw=2, linestyle='-.', label=f'Imputed Messy (AUC = {auc_m:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Predicting Acute Kidney Injury (AKI) - Robustness Test')
    plt.legend(loc="lower right")
    roc_path = os.path.join(plots_dir, "phd_proposal_roc.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4b. Feature Importance (from the Messy model)
    importances = rf_messy.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [features[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.title("RF Feature Importances (Imputed Cohort)")
    bars = plt.bar(range(len(importances)), importances[indices], align="center", color='skyblue')
    plt.xticks(range(len(importances)), feature_names, rotation=45, ha='right')
    plt.xlim([-1, len(importances)])
    plt.ylabel("Relative Importance")
    
    # Highlight Vanco and Zosyn to show the model "discovered" the drug-drug interaction components
    for i, name in enumerate(feature_names):
        if name in ['received_vanco', 'received_zosyn']:
            bars[i].set_color('salmon')

    fi_path = os.path.join(plots_dir, "phd_proposal_feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"-> Saved plots to {plots_dir}")
    
    # 5. Save the Dataset
    csv_clean_path = os.path.join(data_dir, "phd_proposal_synthetic_cohort.csv")
    df.to_csv(csv_clean_path, index=False)
    print(f"-> Saved clean synthetic dataset to {csv_clean_path}")

if __name__ == "__main__":
    main()
