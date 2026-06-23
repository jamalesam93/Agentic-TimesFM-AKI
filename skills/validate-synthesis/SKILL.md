---
name: validate-synthesis
description: Execute, check, and diagnose the statistical validation suite for the EHR Synthesis Engine (Wasserstein Distance, JSD, Chi-Square, Row Collision).
category: Quality Assurance
---

# Validate Synthesis Skill

This playbook guides AI agents and developers in running and troubleshooting the statistical validation suite of the EHR Synthesis Engine.

## Prerequisites

Ensure all dependencies are installed in the Python environment:
```bash
pip install -r requirements.txt
```

## Step-by-Step Execution Playbook

### Step 1: Run the Pipeline to Generate Cohorts
Run a simulation run to write outputs:
```bash
python main.py --n-patients 1000 --n-synthetic 2000 --save-reports 5 --output-dir output
```

### Step 2: Execute the Validation Suite
Compare the synthetic cohort baseline against the raw historical database:
```bash
python -m tests.test_validation --output-dir output --plots-dir plots
```

### Step 3: Analyze the Output
Review the validation console summary:
- Ensure all 6 checks result in a `[PASS]`.
- Check `plots/distribution_fidelity.png` for a visual overlay of Serum Creatinine and Age distributions.

---

## Troubleshooting Validation Failures

If a check fails, follow these diagnostic procedures:

### 1. Wasserstein Distance or Jensen-Shannon Divergence Fails on Baseline SCr
*   **Symptom**: `W1(SCr) >= 0.3` or `JSD(SCr) >= 0.15`.
*   **Cause**: The synthesized baseline SCr distribution does not match the source distribution shape (e.g., rightward shift or excessive noise).
*   **Fix**: 
    1. Confirm that parameters are extracted and synthesized in **log-space** (see `validation_report.md` math).
    2. Check that the variance is correctly extracted and applied: `np.random.normal(0, np.sqrt(log_variance))`.
    3. Verify that the post-noise clipping bounds in `data_extraction.py` and `generator.py` are aligned.

### 2. Chi-Square Vanco+Zosyn Synergy Test Fails
*   **Symptom**: `p-value >= 0.05` (the null hypothesis of independence cannot be rejected).
*   **Cause**: The synthetic cohort lost the statistical signal showing that the Vancomycin + Zosyn co-administration greatly escalates AKI risk.
*   **Fix**:
    1. Check `src/generator.py` -> `synthesize_cohort()`: ensure that `aki_prob[synergy_mask]` is mapped to the extracted parameter `aki_rate_vanco_zosyn`.
    2. Ensure the boolean masking for `synergy_mask` properly selects patients who received *both* drugs.

### 3. Row Collision Check Fails
*   **Symptom**: `collision_rate >= 0.05` (more than 5.0% of synthetic records are identical duplicates of source records).
*   **Cause**: The synthetic cohort is memorizing the source data (under-dispersed sampling).
*   **Fix**:
    1. Check if the seed is fixed or if the noise variance is too low (e.g., if standard deviation is set to zero).
    2. Under Differential Privacy, ensure epsilon isn't too large (which reduces noise).

### 4. Privacy Budget Exceeded Error
*   **Symptom**: `ValueError: Privacy budget exceeded` during data extraction.
*   **Cause**: Sequential composition sum $\sum \epsilon_i$ exceeds the allocated global budget `epsilon`.
*   **Fix**:
    1. Check `src/data_extraction.py` -> `extract_statistical_parameters()`.
    2. Verify that the sum of epsilons spent on Group A, B, and C queries equals exactly the global `epsilon` parameter.
    3. Ensure derived variables (like regression intercept $\beta_0$) are calculated as **post-processing steps** (no extra query and no epsilon consumption).
