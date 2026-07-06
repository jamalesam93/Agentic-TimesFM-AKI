# 3. Methods

## 3.1 Data Sources and Cohort Construction

We constructed a synthetic cohort using a differential privacy (DP) pipeline. The source data consisted of the HDHI Admission Data (providing intensive care unit demographics, baseline serum creatinine (SCr), and comorbidities) and the CKD Nephrotoxic Drug Dataset (providing Vancomycin toxicity and AKI incidence rates). 

We extracted statistical parameters—demographic marginals, log-space SCr distributions, and drug-outcome hazard rates—using a formal $(\epsilon, 0)$-differentially private Laplace mechanism [1]. We sequentially composed the total privacy budget $(\epsilon)$ and partitioned it: 40% for demographics, 35% for SCr distributions, and 25% for drug exposure rates. We derived $L_1$ global sensitivities for bounded means, bounded variances, and ordinary least squares (OLS) regression coefficients to calibrate the Laplace noise.

Using these parameters, we generated a synthetic cohort of 5,000 patients via vectorized sampling. We simulated real-world Electronic Health Record (EHR) noise by injecting age typographical errors (0.5%), SCr measurement outliers (0.5%), and missing data under a Missing Completely at Random (MCAR) assumption [2] for SCr (15%), hypertension (8%), and diabetes (5%).

## 3.2 Longitudinal Trajectory Simulation

We simulated a 5-day longitudinal ICU trajectory for each patient, generating daily Mean Arterial Pressure (MAP) and Vancomycin trough levels. For patients assigned to the AKI outcome—driven by the synergistic toxicity of Vancomycin and Piperacillin-Tazobactam (Zosyn)—SCr values exhibited a delayed rise peaking between days 3 and 5, matching the presentation of acute tubular necrosis (ATN). We mapped peak SCr elevations to KDIGO (Kidney Disease: Improving Global Outcomes) staging criteria [3].

## 3.3 Agentic Multi-Model Architecture

We developed a dual-model agentic framework pairing a Large Language Model (LLM) with a time-series forecasting foundation model.

### 3.3.1 TimesFM Clinical Covariate Forecaster
We fine-tuned Google's `timesfm-2.5-200m-pytorch` [4] for multivariate clinical forecasting using Low-Rank Adaptation (LoRA) weights [5] ($r=16, \alpha=32$) on the attention modules (`qkv_proj`, `out`, `ff0`, `ff1`). The model forecast day 4 and 5 SCr values using a 3-day context of SCr, MAP, Vancomycin troughs, and Zosyn active status.

### 3.3.2 Gemma-4 12B Clinical Sentinel
We fine-tuned `google/gemma-4-12b-it` [6] via QLoRA [7] (4-bit NormalFloat quantization) on 2,000 structured clinical narrative examples derived from the synthetic trajectories. Training used a cosine learning rate schedule over 2 epochs with an effective batch size of 16, adapting seven target modules across attention projections and Multilayer Perceptron (MLP) layers.

### 3.3.3 Agentic Inference
During inference, the Gemma-4 Sentinel receives a 3-day clinical flowsheet. The LLM invokes the TimesFM module to project a 48-hour SCr forecast, applies KDIGO staging logic to the projected values, and synthesizes a structured clinical note to classify the patient's imminent AKI risk.

### 3.3.4 Causal Attention Heatmap Extraction
We evaluated the feature importance and clinical logic of the agent by extracting the raw causal attention weights from the fine-tuned Gemma-4 model. During inference on the holdout dataset, we captured the query and key (Q/K) matrices from the model's self-attention layers. We aggregated the attention weights directed toward specific prior clinical tokens (e.g., Vancomycin trough values, Zosyn status) when generating the final classification token (`[AKI_STAGE_1+]` or `[NORMAL]`). This enabled us to plot literal causal attention heatmaps natively, verifying the model applies pharmacological logic rather than spurious correlations.

## 3.4 Evaluation Protocol and ML Baselines

We evaluated system performance on a held-out dataset of 200 real-world derived trajectories. We trained five machine learning classifiers (Logistic Regression, Random Forest, Support Vector Machine, Gradient Boosting, and XGBoost [8]) on static and 3-day temporal features (age, gender, SCr, MAP, Vanco trough, Zosyn status). 

We evaluated all models under a 25% simulated EHR sparsity condition, imputing missing features with 0 or median values. We report Accuracy, Sensitivity, Specificity, Precision, and F1 Score. We set clinical quality gates at Sensitivity $\ge 0.95$, Specificity $\ge 0.85$, and F1 Score $\ge 0.90$. 

We assessed statistical significance using McNemar's test [9] and computed Cohen's Kappa [10] for inter-rater agreement. We computed 95% confidence intervals for all primary metrics using bootstrapping with 10,000 iterations.

## References
[1] Dwork, C. (2008). Differential Privacy: A Survey of Results. *Theory and Applications of Models of Computation*, 1-19.
[2] Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.
[3] KDIGO. (2012). KDIGO Clinical Practice Guideline for Acute Kidney Injury. *Kidney International Supplements*, 2(1), 1-138.
[4] Das, A., et al. (2024). A decoder-only foundation model for time-series forecasting. *arXiv preprint arXiv:2310.10688*.
[5] Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
[6] Google DeepMind. (2024). Gemma: Open Models Based on Gemini Research and Technology. *Technical Report*.
[7] Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *Advances in Neural Information Processing Systems*.
[8] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
[9] McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.
[10] Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37-46.
