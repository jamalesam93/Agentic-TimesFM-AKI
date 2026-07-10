# Agentic-TimesFM Framework for AKI Prediction

A specialized, privacy-preserving multi-modal framework that integrates a large language model (**Gemma-4 12B**) with a time-series foundation model (**TimesFM 2.5**) to predict synergistic nephrotoxicity (Acute Kidney Injury) induced by Vancomycin and Piperacillin-Tazobactam (Zosyn).

## 🚀 Model Availability
Our finalized model weights and GGUF quants are available on Hugging Face:
[**QinEmPeRoR93/Agentic-TimesFM-AKI-12B**](https://huggingface.co/QinEmPeRoR93/Agentic-TimesFM-AKI-12B)

*(For detailed architecture, training configuration, and deployment instructions, please see the [Hugging Face Model Card](model_cards/README.md)).*

---

## ⚕️ Clinical Context
In intensive care units, co-administration of **Vancomycin** and **Zosyn** is associated with a highly elevated risk of synergistic acute kidney injury (AKI). This framework acts as a clinical sentinel, projecting 72-hour serum creatinine trends and generating interpretable clinical warnings before irreversible renal damage occurs.

---

## 📂 Repository Structure

```text
AKI-training/
├── Article/                 # Final manuscript drafts, figures, and formatting scripts
├── data/                    # Final evaluated datasets
│   ├── raw_source_data/     # Original PhysioNet unorganized CSVs
│   ├── real_world/          # Processed real-world data (eICU, MIMIC-IV)
│   └── synthetic/           # Differentially private synthetic datasets
├── model_cards/             # Hugging Face model documentation
├── reports/                 # Verified evaluation metrics and reports
│   ├── graphs/              # Confusion matrices and ROC curves
│   └── real_world/          # JSON/JSONL metric outputs for eICU and MIMIC-IV
├── scripts/                 # Core reproducible execution scripts
│   ├── real_world/          # Baseline ML runs and real-world holdout evaluation scripts
│   └── synthetic/           # Synthetic cohort generation
├── src/                     # Core python libraries for data extraction & serialization
└── main_real.py             # Master orchestrator for EHR extraction & simulation
```

---

## 📊 Evaluation & Metrics
The framework was trained exclusively on **differentially private synthetic EHR data** (ε = 10) to preserve patient privacy without exposing PHI. It was rigorously validated on a real-world **eICU holdout cohort (N=200)** against traditional ML baselines:

| Model Configuration | Accuracy | Sensitivity | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gemma-4 + TimesFM (Agentic)** | **0.970** | **0.944** | **0.991** | **0.966** |
| Random Forest (Baseline) | 0.807 | 0.709 | 0.877 | 0.752 |
| XGBoost (Baseline) | 0.795 | 0.739 | 0.834 | 0.748 |

> **Warning - Format Fragility**: This model is currently fragile to out-of-domain schema shifts (e.g., when tested directly on MIMIC-IV formats without prompt adaptation). Site-specific prompt calibration and strict schema harmonization are highly recommended before external deployment.

---

## 🛠 Execution & Reproduction

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run ML Baselines:**
   To reproduce the baseline metrics on the synthetic cohort:
   ```bash
   python scripts/real_world/run_ml_baselines.py
   ```

3. **Data Generation Engine:**
   To re-synthesize or extract features via the clinical engine:
   ```bash
   python main_real.py --n-patients 1000 --n-synthetic 50 --epsilon 10
   ```
