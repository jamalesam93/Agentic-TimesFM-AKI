import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import xgboost as xgb

# Set academic styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2
})

def load_data(filepath):
    X = []
    y = []
    with open(filepath, 'r') as f:
        for line in f:
            d = json.loads(line)
            scr = d.get('scr')
            if not scr or len(scr) < 5: continue
            
            baseline_scr = scr[0]
            max_scr = max(scr)
            is_aki = 1 if max_scr >= 1.5 * baseline_scr else 0
            
            features = [
                d['age'],
                d['gender_encoded'],
                scr[0], scr[1], scr[2],
                d['map'][0], d['map'][1], d['map'][2],
                d['vanco_trough'][0], d['vanco_trough'][1], d['vanco_trough'][2],
                d['zosyn_active'][0], d['zosyn_active'][1], d['zosyn_active'][2]
            ]
            
            X.append(features)
            y.append(is_aki)
            
    return np.array(X), np.array(y)

def simulate_ehr_sparsity(X, missing_rate=0.25):
    np.random.seed(42)
    mask = np.random.binomial(1, missing_rate, X.shape)
    X_noisy = X.copy()
    X_noisy[mask == 1] = 0.0 # Missing labs default to 0
    return X_noisy

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root_dir, "data", "real_world", "paper_timesfm_dataset.jsonl")
    
    print(f"Loading real-world data from {data_path}...")
    X, y = load_data(data_path)
    
    # Train test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Inject 25% missing data sparsity
    X_test_noisy = simulate_ehr_sparsity(X_test, missing_rate=0.25)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    colors = ['#2ca02c', '#ff7f0e', '#d62728'] # Green, Orange, Red
    
    plt.figure(figsize=(8, 8))
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    print("\n--- Brier Scores ---")
    for (name, model), color in zip(models.items(), colors):
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test_noisy)[:, 1]
        
        # Calculate Brier score
        brier = brier_score_loss(y_test, y_prob)
        print(f"{name}: {brier:.4f}")
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", color=color, 
                 label=f"{name} (Brier = {brier:.3f})")

    plt.ylabel("Fraction of Positives (Observed Frequency)")
    plt.xlabel("Mean Predicted Probability")
    plt.title('Calibration Curve (Reliability Diagram)\nML Baselines under 25% EHR Sparsity')
    plt.legend(loc="lower right")
    
    plots_dir = os.path.join(root_dir, "plots", "real_world")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "calibration_curve.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved calibration curve to {out_path}")
    
    # Write a small explanation regarding the LLM calibration
    readme_path = os.path.join(plots_dir, "calibration_readme.md")
    with open(readme_path, "w") as f:
        f.write("# Calibration Analysis\n\n")
        f.write("The `calibration_curve.png` shows the reliability diagrams for the traditional ML baselines. ")
        f.write("A Brier score closer to 0 indicates better calibration.\n\n")
        f.write("## Why isn't the LLM on this graph?\n")
        f.write("The LLM generates discrete text classifications (`[AKI_STAGE_1+]` or `[NORMAL]`) rather than ")
        f.write("continuous probabilities. Because the API we use for evaluation does not return log-probabilities ")
        f.write("for the generated tokens, we cannot mathematically construct a continuous calibration curve or ")
        f.write("compute a Brier score for the LLM without modifying the evaluation server to expose logprobs.\n")

if __name__ == "__main__":
    main()
