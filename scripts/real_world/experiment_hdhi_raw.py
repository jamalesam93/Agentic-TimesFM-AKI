import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def main():
    print("==================================================")
    print("    HDHI RAW DATASET EXPERIMENT (NO SYNTHESIS)   ")
    print("==================================================")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hdhi_path = os.path.join(root_dir, "data", "datasets_not_organized", "useful_for_our_case", "HDHI Admission data.csv")
    
    if not os.path.exists(hdhi_path):
        print(f"Error: Could not find HDHI dataset at {hdhi_path}")
        return
        
    print(f"Loading {hdhi_path}...")
    df = pd.read_csv(hdhi_path)
    print(f"Loaded {len(df)} records.\n")
    
    print("Preprocessing raw HDHI data...")
    # Features
    df['age'] = pd.to_numeric(df['AGE'], errors='coerce')
    df['is_male'] = df['GENDER'].astype(str).str.strip().str.upper().str.startswith('M').astype(int)
    df['has_htn'] = pd.to_numeric(df['HTN'], errors='coerce')
    df['has_dm'] = pd.to_numeric(df['DM'], errors='coerce')
    df['baseline_scr'] = pd.to_numeric(df['CREATININE'], errors='coerce')
    
    # Label
    df['developed_aki'] = pd.to_numeric(df['AKI'], errors='coerce')
    
    # Drop rows with missing labels or completely empty features
    df = df.dropna(subset=['developed_aki'])
    
    # Impute missing features with medians
    features = ['age', 'is_male', 'has_htn', 'has_dm', 'baseline_scr']
    for f in features:
        df[f] = df[f].fillna(df[f].median())
        
    X = df[features]
    y = df['developed_aki'].astype(int)
    
    print(f"Training on {len(X)} valid patient records...")
    print(f"Target AKI prevalence: {y.mean()*100:.2f}%\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    print("Evaluating Machine Learning Models on RAW HDHI Data:\n")
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"-> {name} ROC AUC Score: {auc_score:.3f}")

if __name__ == "__main__":
    main()
