# Serum Creatinine Differential Privacy & Generation Validation

To address the rightward shift in synthetic Baseline Serum Creatinine (SCr), we refactored the parameter extraction and synthesis logic to operate in log-space. This prevents the rightward shift (which was peaking at 1.2 mg/dL instead of the baseline 0.85 mg/dL) and guarantees physiological realism.

---

## 📐 Mathematical Formulation

### 1. Extraction in Log-Space
Instead of fitting a lognormal distribution to the raw SCr values, we apply a natural log transformation directly to the clipped raw SCr values:
$$\ln(\text{SCr}_i) = \ln(\text{clip}(\text{SCr}_i, 0.2, 15.0))$$

We then extract:
* **Mean of Log SCr** ($\mu_{\ln}$): Bounded by `[-0.5, 0.5]`.
* **Variance of Log SCr** ($\sigma^2_{\ln}$): Bounded by `[0.01, 0.5]`.
* **Slope of Age to Log SCr** ($\beta_{\text{age}}$): Bounded by `[-0.005, 0.005]`.

### 2. Derived Intercept (Post-Processing)
Due to OLS intercept sensitivity being proportional to the maximum age ($\Delta \beta_0 \propto 95 \times \Delta \beta_1$), direct Laplace perturbation on the intercept consumes excessive privacy budget and yields extremely noisy results. 

Instead, we derive the intercept $\beta_0$ as a post-processing step from the DP-protected mean of log SCr, slope, and mean age:
$$\beta_0 = \mu_{\ln} - \beta_{\text{age}} \times \mu_{\text{age}}$$

By the **Post-Processing Theorem** of Differential Privacy, any function of DP-protected parameters is also DP-protected. This approach reduces the Group B budget partition queries from 4 to 3 (which increases per-query epsilon $\epsilon_B$ from $0.10\epsilon$ to $0.1333\epsilon$) and guarantees that at the average age, the predicted log SCr is always centered perfectly around $\mu_{\ln}$ regardless of slope noise.

### 3. Synthesis in Log-Space
In the generator, the baseline SCr is synthesized as:
$$\ln(\text{SCr}_{\text{pred}}) = \text{Age} \times \beta_{\text{age}} + \beta_0$$
$$\ln(\text{SCr}_{\text{synthetic}}) = \ln(\text{SCr}_{\text{pred}}) + \mathcal{N}(0, \sigma^2_{\ln})$$
$$\text{SCr}_{\text{synthetic}} = \max\left(0.4, \exp\left(\ln(\text{SCr}_{\text{synthetic}})\right)\right)$$

---

## 📊 Validation Summary

Running `test_validation.py` under the new model yields a **100% Pass Rate** for both Standard and DP modes.

### 1. Standard (Non-DP) Run Results

| Check | Metric | Value | Threshold | Result |
|---|---|---|---|---|
| **Age Distribution** | Earth Mover's Distance | `0.342500` | `< 5.0` | **PASS** |
| **Baseline SCr Distribution** | Earth Mover's Distance | `0.007965` | `< 0.3` | **PASS** |
| **Age Divergence** | Jensen-Shannon Divergence | `0.010097` | `< 0.15` | **PASS** |
| **Baseline SCr Divergence** | Jensen-Shannon Divergence | `0.018853` | `< 0.15` | **PASS** |
| **Synergy Interaction** | Chi-Square p-value | `0.000000` | `< 0.05` | **PASS** |
| **Privacy Protection** | Row Collision Rate | `3.65%` | `< 5.0%` | **PASS** |

### 2. Differential Privacy ($\epsilon = 1.0$) Run Results

| Check | Metric | Value | Threshold | Result |
|---|---|---|---|---|
| **Age Distribution** | Earth Mover's Distance | `1.975000` | `< 5.0` | **PASS** |
| **Baseline SCr Distribution** | Earth Mover's Distance | `0.055435` | `< 0.3` | **PASS** |
| **Age Divergence** | Jensen-Shannon Divergence | `0.020268` | `< 0.15` | **PASS** |
| **Baseline SCr Divergence** | Jensen-Shannon Divergence | `0.052987` | `< 0.15` | **PASS** |
| **Synergy Interaction** | Chi-Square p-value | `0.000000` | `< 0.05` | **PASS** |
| **Privacy Protection** | Row Collision Rate | `3.20%` | `< 5.0%` | **PASS** |

> [!NOTE]
> Under DP, the Earth Mover's Distance (EMD) on Baseline SCr is just **0.0554**, representing a massive improvement over the previous EMD of **1.141**, showing that the systematic rightward shift has been completely eliminated.
