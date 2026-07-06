import pandas as pd
from pathlib import Path

data_dir = Path("data/real_world/eicu-collaborative-research-database-demo-2.0.1")

print("--- Patient ---")
df_pat = pd.read_csv(data_dir / "patient.csv")
print(df_pat.columns.tolist())

print("\n--- Lab ---")
df_lab = pd.read_csv(data_dir / "lab.csv")
print("Columns:", df_lab.columns.tolist())
print("Unique lab names matching 'creat':", df_lab[df_lab['labname'].str.contains('creat', case=False, na=False)]['labname'].unique())
print("Unique lab names matching 'vanco':", df_lab[df_lab['labname'].str.contains('vanco', case=False, na=False)]['labname'].unique())

print("\n--- Medication ---")
df_med = pd.read_csv(data_dir / "medication.csv", low_memory=False)
print("Columns:", df_med.columns.tolist())
print("Unique meds matching 'vanco':", df_med[df_med['drugname'].str.contains('vanco', case=False, na=False)]['drugname'].unique()[:5])
print("Unique meds matching 'piperacillin' or 'zosyn':", df_med[df_med['drugname'].str.contains('piperacillin|zosyn', case=False, na=False)]['drugname'].unique()[:5])

print("\n--- Vitals (Aperiodic) ---")
df_vital_aperiodic = pd.read_csv(data_dir / "vitalAperiodic.csv")
print("Columns:", df_vital_aperiodic.columns.tolist())
