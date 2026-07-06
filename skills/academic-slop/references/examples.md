# Before/After Examples: Academic Writing

## Example 1: Vague Significance + Throat-Clearing

**Before:**
> "Acute kidney injury represents a critical gap in the current landscape of patient care. The implications are significant and the consequences are far-reaching. It is well established that AKI contributes to increased morbidity and mortality. A growing body of literature suggests that early detection could improve outcomes."

**After:**
> "Acute kidney injury affects 15-20% of hospitalized patients and increases 30-day mortality 3-5 fold. Each hour of delayed detection raises mortality risk by roughly 2%, yet current monitoring relies on serum creatinine measurements taken every 12-24 hours — a lag that leaves most patients without timely intervention."

**Changes:** Removed vague significance claims. Replaced throat-clearing with specific prevalence and mortality data. Named the actual gap (infrequent monitoring) with evidence.

---

## Example 2: Hedging Abuse + Methodology Theater

**Before:**
> "It is suggested that machine learning approaches may potentially offer improved prediction capabilities for nephrotoxicity. We employed XGBoost, a powerful gradient boosted decision tree framework that has shown promising results across various domains, with carefully tuned hyperparameters including max_depth=6, learning_rate=0.01, n_estimators=500, subsample=0.8, and colsample_bytree=0.8."

**After:**
> "Machine learning improves AKI prediction over clinician judgment alone. We trained XGBoost models with hyperparameters optimized via Bayesian search (100 trials), selecting the configuration that maximized AUPRC on the validation set."

**Changes:** Committed to the claim with evidence ("improves over clinician judgment"). Stripped XGBoost description to essentials. Moved hyperparameters to the implied appendix/table.

---

## Example 3: Gap-Contribution Template + Citation Padding

**Before:**
> "Despite significant advances in AKI prediction, a critical gap remains in the generalizability of these models across diverse patient populations and healthcare systems. This study addresses this gap by developing a comprehensive framework that builds on Smith et al. (2020), Jones et al. (2021), Lee et al. (2022), and Wang et al. (2023) to create a novel approach."

**After:**
> "AKI prediction models achieve internal AUC >0.85 in single-center studies, but external validation across different healthcare systems has been attempted in only 14 of 47 published studies (JAMIA Open, 2025). Of those, performance degrades to AUC 0.70-0.78 on average. We address this by validating our model across seven international cohorts spanning the US, Europe, and Asia."

**Changes:** Named the gap with a specific statistic. Removed citation padding. Stated what the study does in present tense. Made the contribution concrete (seven cohorts).

---

## Example 4: Overstatement + Metaphor Abuse

**Before:**
> "This groundbreaking study represents a paradigm shift in the early detection of medication-induced kidney injury. By bridging the gap between episodic lab monitoring and continuous physiological assessment, this innovative approach paves the way for a new era of proactive renal protection that sheds light on the complex interplay between pharmacological agents and renal function."

**After:**
> "We integrate minute-by-minute physiological data from wearable biosensors with episodic electronic health records to predict nephrotoxic AKI 24-48 hours before serum creatinine criteria. In preliminary validation on synthetic data (n=10,000), the model achieved AUC-ROC of 0.91 with a median prediction lead time of 31 hours."

**Changes:** Removed "groundbreaking," "paradigm shift," "bridges the gap," "paves the way," "sheds light on." Replaced with specific methods, sample size, and quantitative results.

---

## Example 5: Redundant Academic Pairs + Filler Transitions

**Before:**
> "Furthermore, it should be noted that the basic fundamentals of drug-induced nephrotoxicity are closely tied to the final outcomes observed in clinical settings. Additionally, this study will carefully examine the temporal patterns of serum creatinine changes. Moreover, the closely monitored biomarker trajectories provide a brief summary of renal function decline."

**After:**
> "Nephrotoxic drugs impair renal function through direct tubular toxicity, hemodynamic alteration, or crystal deposition. We examine the temporal trajectory of serum creatinine and biomarker changes in the 72 hours preceding AKI onset to identify pre-injury patterns detectable before KDIGO Stage 1 criteria are met."

**Changes:** Removed all filler transitions ("Furthermore," "Additionally," "Moreover"). Cut redundant pairs ("basic fundamentals," "final outcomes," "closely monitor," "brief summary"). Replaced with specific mechanisms and study actions.

---

## Example 6: The Over-Qualified Claim

**Before:**
> "The results of this preliminary analysis may potentially suggest that the integration of continuous monitoring data with electronic health records could, under certain clinical conditions, lead to possible improvements in the early detection of drug-induced kidney injury, although further validation in larger prospective cohorts is warranted."

**After:**
> "In our synthetic validation (n=10,000), adding continuous physiological features to EHR-based models increased AUC-ROC from 0.84 to 0.91 (p<0.001) and extended median prediction lead time from 8 to 31 hours. These results require validation on real-world continuous sensor data."

**Changes:** Replaced hedging with specific numbers. Separated what was shown (synthetic data) from what remains (real-world validation). Made the uncertainty precise.
