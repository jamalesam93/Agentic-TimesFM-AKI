---
language:
- en
license: apache-2.0
tags:
- clinical
- healthcare
- aki
- nephrology
- gguf
- llama.cpp
- gemma
base_model: google/gemma-4-12b-it
---

# Clinical Gemma-4 (12B) - AKI Sentinel [GGUF]

This is the fully merged and quantized (`Q6_K`) GGUF version of the Clinical Gemma-4 (12B) AKI Sentinel model. It was fine-tuned to act as an AI Clinical Pharmacist specializing in predicting and diagnosing Drug-Induced Kidney Disease (DIKD), specifically the synergistic nephrotoxicity caused by **Vancomycin** and **Piperacillin-Tazobactam (Zosyn)**.

## Model Details
- **Architecture:** Gemma-4 (12B Parameters)
- **Format:** GGUF (Quantized to `Q6_K`)
- **Compatibility:** `llama.cpp`, LM Studio, Ollama, text-generation-webui

## Why GGUF?
This format is highly optimized for fast inference on consumer hardware, including Macs with Apple Silicon and standard Windows PCs using CPU/RAM or consumer GPUs.

## Intended Use & Agentic Tool Calling
This LLM has been trained to act within an **Agentic Framework**. It expects a system prompt instructing it to query a time-series model (like TimesFM) for Serum Creatinine (SCr) forecasts before writing a final note. It excels at applying strict KDIGO guidelines to raw creatinine data and drug exposure levels (like Vancomycin troughs).

## Performance
On a holdout set of 200 real-world trajectories (derived from HDHI admissions), this quantized model achieved:
- **Accuracy:** 99.5% (199/200 correct diagnoses)
- **Sensitivity:** 99.0%
- **Specificity:** 100.0%
- **Structured Output Parse Rate:** 100.0%

## Usage with LM Studio / Ollama
1. Download the `.gguf` file.
2. Load it into your local inference server.
3. Use the standard Gemma-4 Chat Template.

**Example Chat Template:**
```xml
<bos><start_of_turn>system
You are a clinical pharmacist. You have access to a TimesFM forecasting tool. Base your assessment on the patient's Day 1-3 data, but ALWAYS query the TimesFM tool to predict Days 4-5 before making a final clinical recommendation.<end_of_turn>
<start_of_turn>user
You are an AI Clinical Pharmacist reviewing a patient's chart at the end of Day 3.
    Patient is actively receiving Vancomycin and Piperacillin-Tazobactam (Zosyn).
    Current SCr trend: Day 1 (1.0), Day 2 (1.1), Day 3 (1.3).
    Vancomycin Trough is rising, currently at 22.0 ug/mL.

    Please assess the risk of nephrotoxicity.<end_of_turn>
<start_of_turn>model
```

## Disclaimer
**For Research Purposes Only.** This AI model is an experimental prototype trained on privacy-preserving real-world clinical data. It is NOT a medical device and should NEVER be used for actual clinical decision-making or patient care.
