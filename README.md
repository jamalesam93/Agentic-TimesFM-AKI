<div align="center">

# Agentic-TimesFM-AKI

### A Dual LLM–Time Series Framework for Predicting Drug-Induced Acute Kidney Injury with Privacy-Preserving Synthetic Data

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97-Model_Hub-yellow.svg)](https://huggingface.co/QinEmPeRoR93/Agentic-TimesFM-AKI)
[![medRxiv Preprint](https://img.shields.io/badge/medRxiv-10.64898%2F2026.07.30.26359271-008080.svg)](https://www.medrxiv.org/content/10.64898/2026.07.30.26359271v1)

</div>

---

## Overview

**Agentic-TimesFM-AKI** is a privacy-preserving, multi-modal clinical framework that integrates a Large Language Model (**Gemma-4 12B**, fine-tuned via QLoRA) with a zero-shot time-series foundation model (**TimesFM 2.5**, fine-tuned via LoRA) to provide continuous, dynamic prediction of synergistic nephrotoxicity — specifically acute kidney injury (AKI) induced by the concurrent administration of **Vancomycin** and **Piperacillin-Tazobactam (Zosyn)**.

The framework is trained exclusively on **differentially private synthetic EHR data** (ε = 10), ensuring zero exposure to Protected Health Information (PHI), and was rigorously validated on real-world holdout cohorts from the publicly accessible **eICU** and **MIMIC-IV** demo databases.

> [!NOTE]
> This repository accompanies the medRxiv preprint: *"Agentic-TimesFM-AKI: A Dual LLM–Time Series Framework for Predicting Drug-Induced Acute Kidney Injury with Privacy-Preserving Synthetic Data"* — Alsakkaf, G. E. A. (2026). medRxiv: [https://www.medrxiv.org/content/10.64898/2026.07.30.26359271v1](https://www.medrxiv.org/content/10.64898/2026.07.30.26359271v1).

---

## Architecture

The framework operates on a **dual-model agentic architecture** where two specialized models collaborate through an orchestration layer:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Agentic Orchestration Layer                     │
│                                                                     │
│   ┌───────────────────────┐       ┌───────────────────────────┐     │
│   │   TimesFM 2.5 Agent   │       │   Gemma-4 12B Sentinel    │     │
│   │   (LoRA Fine-Tuned)   │       │   (QLoRA Fine-Tuned)      │     │
│   │                       │       │                           │     │
│   │  • Ingests 72h labs   │──────▶│  • Receives TimesFM       │     │
│   │  • Projects future    │       │    creatinine projections  │     │
│   │    serum creatinine   │       │  • Integrates demographics │     │
│   │    trajectory         │       │    + medication history    │     │
│   │                       │       │  • Outputs: AKI_POSITIVE  │     │
│   │                       │       │    or AKI_NEGATIVE         │     │
│   │                       │       │  • Generates interpretable │     │
│   │                       │       │    clinical reasoning      │     │
│   └───────────────────────┘       └───────────────────────────┘     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  DP Synthetic Data Engine (ε = 10, Laplace Mechanism)       │   │
│   │  • 5,000 synthetic patient trajectories                     │   │
│   │  • Labs (40%) · Hemodynamics (35%) · Demographics (25%)     │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

1. **TimesFM 2.5 Agent** — A foundation model for time-series forecasting, LoRA fine-tuned to project 72-hour serum creatinine trajectories from longitudinal ICU lab data.
2. **Gemma-4 12B Sentinel** — An instruction-tuned LLM, QLoRA fine-tuned on synthetic clinical narratives. It receives TimesFM's projections alongside patient demographics and medication histories, then synthesizes a binary AKI risk classification with a fully interpretable clinical reasoning chain.
3. **DP Synthetic Data Engine** — A Laplace-mechanism differential privacy pipeline that generates realistic EHR trajectories for training without exposing any real patient data.

---

## Results

### Internal Validation — eICU Holdout (N=200)

The agentic framework significantly outperformed all traditional ML baselines on the real-world eICU holdout cohort:

| Model | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Gemma-4 + TimesFM (Agentic)** | **0.970** | **0.944** | **0.991** | **0.988** | **0.966** |
| Random Forest | 0.807 | 0.709 | 0.877 | 0.801 | 0.752 |
| XGBoost | 0.795 | 0.739 | 0.834 | 0.758 | 0.748 |
| Logistic Regression | 0.713 | 0.812 | 0.643 | 0.615 | 0.700 |

<details>
<summary><b>95% Bootstrap Confidence Intervals (n = 10,000 resamples)</b></summary>

| Metric | Point Estimate | 95% CI |
|:---|:---:|:---:|
| Accuracy | 0.970 | 0.945 – 0.990 |
| Sensitivity | 0.944 | 0.892 – 0.988 |
| Specificity | 0.991 | 0.971 – 1.000 |
| Precision | 0.988 | 0.961 – 1.000 |
| F1-Score | 0.966 | 0.934 – 0.989 |

</details>

<details>
<summary><b>Statistical Significance (McNemar's Test)</b></summary>

All pairwise comparisons against the agentic framework were statistically significant at p < 0.01:

| Comparison | χ² | p-value |
|:---|:---:|:---:|
| vs. Logistic Regression | 31.03 | < 0.001 |
| vs. SVM (Linear) | 37.03 | < 0.001 |
| vs. Random Forest | 20.05 | < 0.001 |
| vs. XGBoost | 12.50 | < 0.001 |
| vs. Gradient Boosting | 16.41 | < 0.001 |

</details>

### External Validation — MIMIC-IV Holdout (N=117)

| Metric | Value |
|:---|:---:|
| Accuracy | 0.718 |
| Sensitivity | 0.750 |
| Specificity | 0.713 |
| F1-Score | 0.421 |

> [!WARNING]
> External validation revealed severe performance degradation due to **structural formatting fragility** and **domain shift** between eICU and MIMIC-IV schema formats. The model is currently not generalizable across sites without schema harmonization and prompt recalibration. This limitation is discussed transparently in the manuscript.

---

## Repository Structure

```text
Agentic-TimesFM-AKI/
│
├── src/                                # Core Python libraries
│   ├── data_extraction_real.py         # EHR data extraction & serialization
│   ├── data_ingestion_loader.py        # Data loading utilities
│   ├── finetune_timesfm.py             # TimesFM LoRA fine-tuning
│   ├── generator.py                    # DP synthetic data generation engine
│   ├── hyperparameter_tuning.py        # Hyperparameter search
│   ├── sentinel_inference.py           # Gemma-4 inference pipeline
│   └── textualization.py               # EHR → natural language conversion
│
├── scripts/
│   ├── train_qlora_gemma4_12b.py       # Gemma-4 QLoRA training script
│   ├── merge_adapter.py                # LoRA adapter merging
│   ├── extract_attention_heatmaps.py   # Attention visualization
│   ├── run_agent_pipeline.py           # End-to-end agentic evaluation
│   ├── real_world/                     # Real-world evaluation scripts
│   │   ├── experiment_real.py          # Main eICU evaluation
│   │   ├── run_ml_baselines.py         # ML baseline comparison
│   │   ├── bootstrap_confidence_intervals.py
│   │   ├── statistical_significance.py
│   │   ├── calibration_analysis.py
│   │   └── varying_epsilon_analysis.py
│   └── synthetic/
│       └── experiment_synthetic.py     # Synthetic cohort evaluation
│
├── data/
│   ├── paper_sft_dataset.jsonl         # SFT training dataset (synthetic)
│   ├── eicu_eval_holdout.jsonl         # eICU holdout (N=200)
│   ├── mimic_eval_holdout.jsonl        # MIMIC-IV holdout (N=117)
│   ├── raw_source_data/                # Original PhysioNet CSVs
│   ├── real_world/                     # Processed real-world data
│   └── synthetic/                      # DP synthetic datasets
│
├── reports/
│   ├── graphs/                         # Confusion matrices, ROC curves
│   └── real_world/                     # JSON metric outputs & CI reports
│
├── model_cards/
│   └── README.md                       # Hugging Face model card
│
├── main_real.py                        # Master orchestrator
└── requirements.txt                    # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (24 GB+ VRAM recommended for full inference; GGUF quantized variants available for consumer hardware)

### Installation

```bash
git clone https://github.com/jamalesam93/Agentic-TimesFM-AKI.git
cd Agentic-TimesFM-AKI
pip install -r requirements.txt
```

### Downloading Model Weights

Pre-trained weights are hosted on the Hugging Face Model Hub:

```bash
# Using the Hugging Face CLI
huggingface-cli download QinEmPeRoR93/Agentic-TimesFM-AKI --local-dir ./models
```

The repository includes:
- `model.safetensors` — Merged Gemma-4 12B with LoRA adapters
- `gguf/` — Quantized GGUF variants for consumer hardware
- `llm_lora_adapter/` — Standalone Gemma-4 QLoRA adapter
- `timesfm_lora_adapter/` — Standalone TimesFM LoRA adapter

---

## Reproduction Guide

### 1. Reproduce ML Baselines

Run traditional ML baselines (Random Forest, XGBoost, Logistic Regression, SVM, Gradient Boosting) on the eICU holdout:

```bash
python scripts/real_world/run_ml_baselines.py
```

### 2. Regenerate Synthetic Data

Re-synthesize the differentially private training cohort with the Laplace mechanism:

```bash
python main_real.py --n-patients 1000 --n-synthetic 50 --epsilon 10
```

### 3. Run the Agentic Evaluation Pipeline

Evaluate the full Gemma-4 + TimesFM pipeline on the eICU holdout:

```bash
python scripts/real_world/experiment_real.py
```

### 4. Bootstrap Confidence Intervals

```bash
python scripts/real_world/bootstrap_confidence_intervals.py
```

### 5. Statistical Significance Tests

```bash
python scripts/real_world/statistical_significance.py
```

---

## Model Card

For detailed model architecture, training configuration, quantization details, and deployment instructions, see the full [Hugging Face Model Card](https://huggingface.co/QinEmPeRoR93/Agentic-TimesFM-AKI).

---

## Known Limitations

| Limitation | Description |
|:---|:---|
| **Formatting Fragility** | The LLM is highly sensitive to input schema variations. Out-of-distribution EHR formats (e.g., MIMIC-IV) cause severe performance degradation without prompt recalibration. |
| **Demo-Scale Evaluation** | Validation was performed on the publicly accessible demo subsets of eICU (N=200) and MIMIC-IV (N=117), not the full credentialed databases. |
| **Token Truncation** | Strict generation limits (e.g., 100 tokens) can artificially truncate the clinical reasoning chain, causing forced false negatives. Dynamic token allocation is recommended. |
| **Single Drug Pair** | Currently validated only for Vancomycin + Piperacillin-Tazobactam nephrotoxicity. Generalization to other nephrotoxic agents requires additional fine-tuning. |

---

## Ethics & Data Access

This study utilized **publicly available, de-identified demo datasets** (eICU Collaborative Research Database Demo, MIMIC-IV Clinical Database Demo) and **differentially private synthetic data**. No credentialing, CITI training, or institutional Data Use Agreements are required to reproduce the results presented.

Scaling to the complete eICU and MIMIC-IV databases requires separate credentialing through [PhysioNet](https://physionet.org) under their respective Data Use Agreements.

---

## Citation

If you use this framework, code, or model weights in your research, please cite:

```bibtex
@article{alsakkaf2026agentictimesfmaki,
  title   = {Agentic-TimesFM-AKI: A Dual LLM–Time Series Framework for Predicting 
             Drug-Induced Acute Kidney Injury with Privacy-Preserving Synthetic Data},
  author  = {Alsakkaf, Gamal Esam Ahmed},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.07.30.26359271},
  url     = {https://www.medrxiv.org/content/10.64898/2026.07.30.26359271v1}
}
```

---

## License

This project is licensed under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

Model weights inherit the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) from the base Gemma-4 model.

---

<div align="center">

**Antigravity Advanced Research Institute**

</div>
