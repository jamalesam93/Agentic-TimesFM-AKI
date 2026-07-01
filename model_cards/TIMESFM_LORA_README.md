---
language:
- en
license: apache-2.0
tags:
- time-series
- forecasting
- clinical
- healthcare
- aki
- peft
- timesfm
base_model: google/timesfm-2.5-200m-pytorch
---

# TimesFM 2.5 - Clinical AKI Covariate Forecaster [LoRA Adapter]

This model is a PEFT (LoRA) adapter specifically fine-tuned on top of Google's foundational time-series model, `timesfm-2.5-200m-pytorch`. It has been adapted to forecast Serum Creatinine (SCr) trajectories by interpreting dynamic clinical covariates that trigger Acute Kidney Injury (AKI).

## Model Details
- **Architecture:** TimesFM 2.5 (200M Parameters)
- **Format:** PEFT (LoRA) Adapter
- **Target Task:** Multivariate Time-Series Forecasting
- **Covariates Learned:** Mean Arterial Pressure (MAP), Vancomycin Troughs, Piperacillin-Tazobactam (Zosyn) Status.

## Why this model was trained
Out of the box (zero-shot), foundational forecasting models like TimesFM do not understand pharmacology. When a patient's SCr is stable for 3 days, standard TimesFM will predict a flat line for the future. 

However, when a patient is receiving concurrent Vancomycin and Zosyn, there is a known synergistic risk of delayed nephrotoxicity. This adapter was trained on thousands of privacy-preserving patient trajectories mathematically derived from real-world parameters (HDHI Admission Data & CKD Nephrotoxic Drug Datasets). It maps the relationship between rising Vancomycin troughs + Zosyn exposure and the resulting delayed spikes in Serum Creatinine.

## Performance
- **Validation MAE:** 0.257 mg/dL on predicting Day 4-5 SCr spikes (evaluated on 1,000 holdout real-world trajectories).

## How to use
Because TimesFM is fundamentally a transformer, we attached standard LoRA weights to its attention modules (`qkv_proj`, `out`, `ff0`, `ff1`). 

```python
import torch
import timesfm
import numpy as np
from peft import PeftModel

# 1. Load Base TimesFM
model_path = "google/timesfm-2.5-200m-pytorch"
base_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path)

# 2. Attach these LoRA weights
adapter_path = "jamalesam93/phd-timesfm-2.5-aki-lora"
pytorch_module = getattr(base_model, 'model', base_model)
pytorch_module = PeftModel.from_pretrained(pytorch_module, adapter_path)
base_model.model = pytorch_module

# Ensure tensors are packed according to the 63-length sequence configuration used during training.
```

## Disclaimer
**For Research Purposes Only.** This AI model is an experimental prototype trained on privacy-preserving real-world clinical data. It is NOT a medical device and should NEVER be used for actual clinical decision-making or patient care.
