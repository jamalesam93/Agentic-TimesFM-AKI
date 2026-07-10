---
language: en
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
license: apache-2.0
datasets:
- synthetic-ehr
metrics:
- accuracy
- f1
- precision
- recall
---

# Agentic-TimesFM-AKI-12B

## Model Description
**Agentic-TimesFM-AKI-12B** is a specialized, agentic multi-modal framework designed for the continuous prediction of synergistic nephrotoxicity, specifically targeting acute kidney injury (AKI) induced by the concurrent administration of Vancomycin and Piperacillin-Tazobactam (Zosyn). 

This model integrates a large language model (**Gemma-4 12B**) with a state-of-the-art time-series forecasting foundation model (**TimesFM 2.5**). It leverages the clinical reasoning capabilities of Gemma-4 alongside TimesFM's precise projection of longitudinal laboratory trends (e.g., serum creatinine) to achieve high precision and interpretability.

## Model Architecture
The framework operates on a dual-model architecture:
1. **TimesFM 2.5 Agent**: Explicitly forecasts patient serum creatinine trajectories over a 72-hour window.
2. **Gemma-4 12B (Instruction-Tuned)**: Receives the TimesFM projections, patient demographics, and medication histories. It acts as the orchestrator to synthesize the data and generate both a binary risk classification and a highly interpretable clinical warning summary.

## Training Data & Privacy
The framework was trained exclusively on **differentially private synthetic EHR data**.
- **Cohort Size:** 2,000 synthetic patient trajectories (823 AKI / 1177 Normal).
- **Privacy Mechanism:** Laplace mechanism with a strict privacy budget of ε = 10, distributed across longitudinal laboratory trends (40%), hemodynamic parameters (35%), and static demographics (25%).
- This ensures patient privacy (no PHI exposure) while retaining the critical multivariate distributions required for the LLM to learn the synergistic nephrotoxicity of the drugs.

## Evaluation Results
The framework underwent rigorous internal validation on a real-world **eICU holdout cohort (N=200)**. It significantly outperformed traditional machine learning baselines:

| Metric | Point Estimate | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | 0.970 | 0.945 – 0.990 |
| **Sensitivity (Recall)** | 0.944 | 0.892 – 0.988 |
| **Specificity** | 0.991 | 0.971 – 1.000 |
| **Precision** | 0.988 | 0.961 – 1.000 |
| **F1-Score** | 0.966 | 0.934 – 0.989 |

*Note: The system features "safe failure" modes. Even when the model yields a false negative due to label misalignment, it consistently generates an alarming, interpretable clinical warning to alert the physician.*

## Limitations and Generalizability
* **Formatting Fragility:** The model currently exhibits severe degradation when subjected to out-of-domain external validation (e.g., MIMIC-IV cohort, F1 collapsed to 0.421). It is highly susceptible to formatting fragility and structural schema shifts. 
* **Deployment Warning:** This model requires strict schema harmonization and site-specific prompt calibration before deployment across disparate hospital networks.
* **Token Limits:** Imposing strict generation limits (e.g., 100 tokens) can artificially truncate the agent's clinical reasoning chain, resulting in forced false negatives. Dynamic token allocation is recommended.

## Usage
The repository provides:
1. **`model.safetensors`**: The merged Gemma-4-12B model weights.
2. **`gguf/`**: Quantized GGUF versions of the model for consumer hardware inference.
3. **`llm_lora_adapter/` & `timesfm_lora_adapter/`**: The standalone adapters if you wish to apply them to your own base models.

### Quick Start (Pseudo-code)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "QinEmPeRoR93/Agentic-TimesFM-AKI-12B"

# Load the LLM Agent
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = """<|turn>user
Patient: 68yo, MAP: 75, Vanco Trough: 18mg/L, Zosyn: Active.
TimesFM 72h Creatinine Projection: [1.2, 1.4, 1.9]
Assess synergistic nephrotoxicity risk.
<|turn>model
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=250)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation
If you use this model or framework, please cite the corresponding manuscript:
*(Citation details pending publication).*
