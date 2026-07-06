# Academic Structures to Avoid

## The Gap-Contribution Template

Many AI-generated academic texts follow this rigid structure:

> "Despite advances in X, a significant gap remains in Y. This study addresses this gap by..."

This is acceptable once in an introduction. If every section uses this template, the writing becomes formulaic.

**Instead:** Vary how you frame gaps. Sometimes lead with the specific limitation. Sometimes lead with a contradictory finding. Sometimes lead with a clinical scenario.

| Formulaic | Varied |
|-----------|--------|
| "Despite advances in X, a gap remains in Y" | "Current models for X achieve AUC >0.85 in single-center studies, but performance degrades to <0.70 in external cohorts" |
| "This study addresses this gap by" | "We extend this work by testing across 7 international cohorts" |
| "Little attention has been paid to" | "Only 2 of 47 published studies have examined..." (cite) |

## Laundry-List Objectives

Four or more objectives listed as a numbered sequence where items 3 and 4 are padding:

> 1. Develop a prediction model
> 2. Validate on external cohorts
> 3. Evaluate feature importance using SHAP
> 4. Design a clinical decision support framework
> 5. Assess algorithmic fairness across demographic groups
> 6. Investigate temporal drift in model performance

If objectives 5 and 6 are not core to the research question, they inflate the proposal. Three strong, specific objectives beat six vague ones.

**Instead:** Limit to 3-4 objectives, each tied to a research question. Name the specific outcome metric for each.

## The "Builds On" Stack

Piling citations to show comprehensive knowledge without integrating the findings:

> "This work builds on Smith et al. (2020), Jones et al. (2021), Lee et al. (2022), Wang et al. (2023), and Chen et al. (2024) to develop a novel framework..."

The reader learns nothing about what these studies found or how they relate.

**Instead:** Name the specific contribution from each study: "This work extends Lee et al.'s attention-based LSTM (2022) by incorporating continuous physiological features that their episodic lab-value model could not process."

## The Weak Contrast

Using negation to create false novelty:

> "Unlike previous studies that focused solely on serum creatinine, this approach also considers..."

If prior work truly only used sCr, state it with a citation. If it did not, the contrast is dishonest.

**Rule:** Every contrast must be verifiable from the cited literature. If you cannot find the citation, do not make the claim.

## The Recursive Abstract

Abstracts that repeat the same information in introduction, methods, and conclusion:

> "Background: AKI is a major problem. Methods: We developed a model for AKI. Results: Our model predicts AKI. Conclusion: Our model addresses AKI."

Each section must add new information not present in other sections.

## The Over-Qualified Claim

Hedging a claim so heavily it becomes meaningless:

> "The results may potentially suggest that the proposed approach could, under certain conditions, lead to possible improvements in early detection of kidney injury, although further research is warranted."

**Instead:** "The proposed approach detected AKI 12 hours before serum creatinine criteria in 78% of cases (95% CI: 72-84%). Validation on independent cohorts is needed."

## Methodological Overdescription

Describing every parameter of a standard tool:

> "We employed XGBoost (Chen & Guestrin, 2016), a gradient boosted decision tree framework, with max_depth=6, learning_rate=0.01, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, and gamma=0."

**Instead:** "We trained XGBoost models with hyperparameters optimized via Bayesian search (100 trials)."

Move the full parameter table to an appendix.

## The "Significance" Paragraph Template

Discussion sections often end with:

> "These findings have significant implications for clinical practice. They shed light on the importance of early detection and pave the way for improved patient outcomes."

**Instead:** "Detecting AKI 12 hours before serum creatinine criteria gives clinicians a treatment window to adjust nephrotoxic dosing or initiate fluid resuscitation, which Yang et al. (2023) showed reduces Stage 3 AKI progression by 31%."

## Numbered Citation Padding

> "Previous studies have investigated this problem [1,2,3,4,5,6,7,8,9,10,11,12,13,14]."

**Instead:** Cite the 2-3 most relevant studies and explain what they found: "Three meta-analyses have examined AKI prediction performance: Aires et al. (2023) reported pooled AUC of 0.82, while Ihsan et al. (2026) found neural networks achieved a median AUROC of 0.90 but noted insufficient external validation in 75% of studies."

## The "To the Best of Our Knowledge" Hedge

> "To the best of our knowledge, this is the first study to..."

This phrase is acceptable once in an abstract or introduction. Using it more than once signals either poor literature review or inflated novelty claims.

**Rule:** Use at most once per document. Verify the claim by searching the literature.
