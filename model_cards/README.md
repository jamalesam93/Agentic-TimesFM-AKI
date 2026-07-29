---
language: en
license: apache-2.0
tags:
- medical
- healthcare
- time-series
- llama-cpp
- gguf
- safetensors
- gemma-4
- timesfm
- aki
- nephrotoxicity
datasets:
- synthetic-ehr
metrics:
- accuracy
- f1
- precision
- recall
- sensitivity
- specificity
base_model: google/gemma-4-12b-it
library_name: transformers
pipeline_tag: text-generation
---

# Agentic-TimesFM-AKI

<div align="center">

### A Dual LLM–Time Series Framework for Predicting Drug-Induced Acute Kidney Injury with Privacy-Preserving Synthetic Data

[![GitHub Repo](https://img.shields.io/badge/GitHub-Agentic--TimesFM--AKI-blue?logo=github)](https://github.com/jamalesam93/Agentic-TimesFM-AKI)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## 📌 Model Overview

**Agentic-TimesFM-AKI** is a specialized, privacy-preserving multi-modal clinical framework designed for the continuous prediction of synergistic nephrotoxicity — specifically Acute Kidney Injury (AKI) induced by the concurrent administration of **Vancomycin** and **Piperacillin-Tazobactam (Zosyn)**.

This repository hosts pre-trained weights, adapters, and quantized GGUF variants for the dual-model system, integrating:
1. **Gemma-4 12B Sentinel** (QLoRA fine-tuned on synthetic clinical narratives).
2. **TimesFM 2.5 Agent** (LoRA fine-tuned zero-shot time-series forecaster).

---

## 🏗️ Architecture & Functionality

The system leverages a dual-agent orchestration framework:
- **TimesFM 2.5 Forecast Engine:** Ingests longitudinal lab values (e.g., serum creatinine, BUN) and projects future 72-hour creatinine trajectories.
- **Gemma-4 12B Clinical Sentinel:** Receives TimesFM's 72-hour projections alongside patient demographics and medication timelines to output a structured binary prediction (`AKI_POSITIVE` / `AKI_NEGATIVE`) and a natural language clinical warning.

```text
Patient EHR / Labs ──▶ TimesFM 2.5 Agent (72h Forecast)
                              │
                              ▼
Patient Context ──────▶ Gemma-4 12B Sentinel ──▶ Binary Risk + Clinical Warning
```

---

## 📊 Evaluation & Metrics

The framework was trained exclusively on **differentially private synthetic data** ($\varepsilon = 10$) to preserve patient privacy and validated on a real-world **eICU holdout cohort (N=200)**:

| Metric | Point Estimate | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **0.970** | 0.945 – 0.990 |
| **Sensitivity (Recall)** | **0.944** | 0.892 – 0.988 |
| **Specificity** | **0.991** | 0.971 – 1.000 |
| **Precision** | **0.988** | 0.961 – 1.000 |
| **F1-Score** | **0.966** | 0.934 – 0.989 |

### Performance Comparison vs. Baselines (eICU Holdout)

- **Agentic-TimesFM-AKI (F1: 0.966)** significantly outperformed traditional baselines:
  - Random Forest (F1: 0.752, $p < 0.001$)
  - XGBoost (F1: 0.748, $p < 0.001$)
  - Logistic Regression (F1: 0.700, $p < 0.001$)

---

## ⚠️ Important Considerations & Limitations

1. **Formatting Fragility (Domain Shift):** External validation on the MIMIC-IV demo cohort revealed performance degradation ($F1 = 0.421$) due to structural schema shifts. Prompt recalibration and strict schema harmonization are required before multi-center deployment.
2. **Generation Token Allocations:** Truncating generation tokens (e.g., `< 100` tokens) prematurely cuts off the clinical reasoning chain, causing false negatives. Dynamic or high token limits ($\ge 250$ tokens) are recommended.
3. **Clinical Scope:** Currently optimized specifically for Vancomycin + Piperacillin-Tazobactam synergistic risk assessment.

---

## 💻 Quick Start & Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "QinEmPeRoR93/Agentic-TimesFM-AKI"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = """<|turn>user
Patient: 68yo male, MAP: 75 mmHg, Vancomycin Trough: 18 mg/L, Piperacillin-Tazobactam: Active.
TimesFM 72h Serum Creatinine Projection: [1.2, 1.4, 1.9 mg/dL]
Assess synergistic nephrotoxicity risk.
<|turn>model
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=250)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ⚖️ Licensing & Terms

- **Code & Model Weights:** Released under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).
- **Base Model License:** Fine-tuned from Google's Gemma 4. Usage must conform with [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

## 📄 Citation

If you use this model or code in your work, please cite the corresponding paper:

```bibtex
@article{saka2026agentictimesfmaki,
  title   = {Agentic-TimesFM-AKI: A Dual LLM–Time Series Framework for Predicting Drug-Induced Acute Kidney Injury with Privacy-Preserving Synthetic Data},
  author  = {Saka, Jamal E.},
  year    = {2026},
  note    = {Manuscript under review}
}
```
