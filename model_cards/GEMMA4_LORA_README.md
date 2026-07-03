---
language:
- en
license: apache-2.0
tags:
- clinical
- healthcare
- aki
- nephrology
- peft
- unsloth
- qlora
- gemma
base_model: google/gemma-4-12b-it
---

# Clinical Gemma-4 (12B) - AKI Sentinel [LoRA Adapter]

This model is a PEFT (LoRA) adapter specifically fine-tuned on top of `google/gemma-4-12b-it` to act as an AI Clinical Pharmacist. It specializes in predicting and diagnosing Drug-Induced Kidney Disease (DIKD), specifically the synergistic nephrotoxicity caused by the co-administration of **Vancomycin** and **Piperacillin-Tazobactam (Zosyn)**.

## Model Details
- **Architecture:** Gemma-4 (12B Parameters)
- **Format:** PEFT (LoRA) Adapter
- **Training Method:** QLoRA via Unsloth (4-bit quantization during training)
- **Target Task:** Clinical note synthesis, AKI KDIGO staging, and dynamic tool-use.

## Intended Use & Agentic Tool Calling
This LLM has been trained to act within an **Agentic Framework**. When reviewing a patient's chart, it is instructed to pause and query an external time-series forecasting model (like TimesFM) for 48-hour Serum Creatinine (SCr) projections before generating its final clinical note. 

The LLM is highly adept at:
1. Synthesizing complex clinical pharmacology trajectories.
2. Mathematically calculating KDIGO AKI thresholds based on baseline SCr.
3. Incorporating tool-responses (like TimesFM SCr predictions) directly into its clinical rationale.

## Evaluation Metrics (Real-World Holdout)
Evaluated on 200 privacy-preserving real-world trajectories (HDHI admissions):
* **Accuracy:** 99.5%
* **Sensitivity (Recall):** 99.0%
* **Specificity (True Negative Rate):** 100.0%
* **F1 Score:** 99.5%

*Note: High sensitivity is achieved specifically because this model was trained to ingest the forecasts from the companion `QinEmPeRoR93/phd-timesfm-2.5-aki-lora` adapter.*

## How to use
This is a LoRA adapter. You must load the base model and apply these weights using the `peft` library.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "google/gemma-4-12b-it"
adapter_name = "QinEmPeRoR93/phd-gemma-4-12b-aki-lora"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
```

## Disclaimer
**For Research Purposes Only.** This AI model is an experimental prototype trained on privacy-preserving real-world clinical data. It is NOT a medical device and should NEVER be used for actual clinical decision-making or patient care without strict human supervision and extensive clinical validation.
