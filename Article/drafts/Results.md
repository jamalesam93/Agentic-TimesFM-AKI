# 4. Results

## 4.1 Differentially Private Synthetic Data Utility
To establish the efficacy of our Laplace-driven Differential Privacy (DP) module, we evaluated the utility tradeoff of synthetic patient trajectories at varying privacy budgets ($\epsilon$). A Random Forest classifier (baseline AUC 0.904 on non-private raw data) was trained on synthetic datasets generated across different strictness levels ($\epsilon \in \{0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$) and subsequently evaluated on the real-world holdout test set. 

At extremely strict privacy regimes ($\epsilon = 0.5$), utility dropped significantly (AUC 0.781), as expected due to heavy noise injection overpowering demographic and laboratory signals. However, at a moderate and highly acceptable privacy budget of $\epsilon = 1.0$, the model regained significant predictive power, achieving an AUC of 0.868. Notably, expanding the budget to $\epsilon = 5.0$ yielded an AUC of 0.918, indicating that our allocated budget partitioning (40% to Labs/Vitals, 35% to Comorbidities/Meds, 25% to Demographics) successfully preserved critical multivariate associations without compromising the privacy bounds.

## 4.2 Machine Learning Baselines vs. Agentic LLM Performance
Using the synthesized EHR trajectories, we established traditional ML baselines against our proposed DIKD Agentic framework. The ML models produced solid but limited discriminative performance:
*   **Random Forest (RF)**: AUC 0.931, Brier Score 0.107
*   **XGBoost (XGB)**: AUC 0.889, Brier Score 0.154
*   **Logistic Regression (LR)**: AUC 0.701

While RF demonstrated high general discrimination, its static feature reliance inherently limits explainability and struggles with longitudinal trajectory shifts. 

By contrast, the **Gemma-4 12B Safety Sentinel** agent achieved a remarkable **92.5% classification accuracy** (185/200 correct predictions) on the holdout evaluation. Unlike the ML baselines, the LLM achieved this while producing transparent, KDIGO-anchored clinical synthesis notes, identifying complex multidrug interactions (e.g., Vancomycin and Piperacillin/Tazobactam synergy) and applying formal definitions of Acute Kidney Injury.

## 4.3 Ablation Study: Impact of Tool-Augmentation
To isolate the value of the agentic tool-use architecture, we conducted an ablation study by stripping the framework of the `TimesFM 2.5` trajectory forecasting tool, restricting the LLM's context exclusively to historical and current measurements (Days 1–3). 

The removal of the predictive forecasting module led to a catastrophic degradation in performance. The LLM's accuracy collapsed from 92.5% to **48.0%**. Furthermore, the model suffered a complete structural adherence failure on 104 out of the 200 evaluation cases (a 48.0% parse failure rate). Stripped of the structural evidence provided by the external tools, the LLM hallucinated arbitrary clinical categorizations (e.g., outputting `[LOW]` instead of KDIGO definitions). This result definitively proves that the robust performance of the DIKD system relies heavily on the synergistic coupling of the LLM's reasoning engine with specialized, deterministic clinical forecasting tools.
