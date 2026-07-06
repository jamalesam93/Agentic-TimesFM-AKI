# Agentic LLM AKI gap report

##### [**Undermind**](https://undermind.ai)

---


## Table of Contents

- [Agentic LLM framework for AKI prediction](#agentic-llm-framework-for-aki-prediction)
  - [Literature map](#literature-map)
  - [What broader AKI work establishes](#what-broader-aki-work-establishes)
  - [Why drug-induced AKI is the best target use case](#why-drug-induced-aki-is-the-best-target-use-case)
  - [Research gap](#research-gap)
  - [Implications for a proposed framework](#implications-for-a-proposed-framework)
  - [Evaluation plan](#evaluation-plan)
  - [Bottom line](#bottom-line)
  - [References](#references)

# Agentic LLM framework for AKI prediction

Acute kidney injury prediction already has a strong quantitative base in longitudinal EHR data, from large continuous prediction systems (Tomašev et al., 2019) to ICU-focused multimodal forecasting (Tan et al., 2024), KDIGO-aware benchmarking (Lyu et al., 2024), and externally validated variable-length time-series models (Bashiri et al., 2024). Drug-induced AKI is narrower but clinically attractive because it is actionable at the point of prescribing, monitoring, and deprescribing, and because drugs are among the more preventable causes of AKI in hospital care (Kom et al., 2023; Mehta et al., 2015).

The main gap is not the absence of predictive models. The gap is the absence of systems that let an agentic LLM inspect an evolving patient trajectory, choose or query the right quantitative renal-risk model, ground its reasoning in protocol logic, and return an auditable recommendation. Clinical agent papers show that tool-using LLMs can retrieve and execute validated risk tools (Jin et al., 2025; Liu et al., 2025), and EHR agents can reason over structured records through code and database tools (W. Shi et al., 2024). But AKI-specific agent papers remain sparse (T. Shi et al., 2024) or closer to structure-following deliberation than true external model orchestration (Gordon et al., 2025). Systems that wrap expert models exist (Wang et al., 2024), yet they usually interpret fixed model outputs rather than dynamically route among renal-risk models. Protocol-constrained longitudinal reasoning is emerging (Qu & Farber, 2026), but it has not yet been tied tightly to nephrotoxicity prediction.

## Literature map

| Layer | What the literature already supports | Main limit for this project | Representative papers |
|:---|:---|:---|:---|
| Longitudinal AKI prediction | Continuous and ICU-based forecasting from structured time series is mature, with strong work on lead time, multimodal inputs, and external validation (Bashiri et al., 2024; Tan et al., 2024; Tomašev et al., 2019). | These papers predict AKI well, but they do not add an LLM orchestration layer. | (Bashiri et al., 2024; Lyu et al., 2024; Tan et al., 2024; Tomašev et al., 2019) |
| Drug-induced and nephrotoxic AKI | Exposure-aware prediction is feasible in both multicenter and hospital-specific settings, including nephrotoxin cohorts and drug-specific models (Ambe et al., 2025; Griffin et al., 2024; Heo et al., 2023; Mu et al., 2022; Okawa et al., 2022). | The literature is fragmented by drug, cohort, and label definition. | (Ambe et al., 2025; Griffin et al., 2024; Heo et al., 2023; Mu et al., 2022; Okawa et al., 2022) |
| Tool-using clinical LLMs | LLM agents can retrieve validated calculators, extract parameters, execute tools, and summarize quantitative outputs (Jin et al., 2025; Liu et al., 2025). | The tools are mostly generic risk calculators rather than longitudinal renal models over raw EHR streams. | (Jin et al., 2025; Liu et al., 2025) |
| EHR-facing agent infrastructure | Agents can query structured records, execute code, debug plans, and reason over temporal EHR queries (W. Shi et al., 2024). Protocol-constrained state tracking is becoming more realistic for long trajectories (Qu & Farber, 2026). | This infrastructure is not yet specialized for nephrotoxicity prediction or model routing. | (Qu & Farber, 2026; W. Shi et al., 2024) |
| Multi-agent clinical interpretation | LLM agents can sit on top of expert models and turn model outputs into consensus-style reports (Wang et al., 2024). | This is closer to interpretation of fixed outputs than dynamic choice among renal prediction tools. | (Gordon et al., 2025; Wang et al., 2024) |
| Privacy-preserving synthetic EHR | Differentially private synthetic sequence generation is real and useful for infrastructure, especially when privacy claims need formal grounding (Lee et al., 2020; Torfi et al., 2020). | Existing privacy papers are not the core novelty here and often simplify temporal or feature realism relative to AKI prediction needs. | (Lee et al., 2020; Torfi et al., 2020) |

## What broader AKI work establishes

Broader AKI prediction work shows that the quantitative problem is already tractable. Large-scale continuous prediction with clinically meaningful lead time was established early (Tomašev et al., 2019). ICU work then pushed toward richer labeling and more realistic data use. A key correction is that the full KDIGO definition materially changes the task because urine output captures many events that creatinine-only labels miss (Lyu et al., 2024). More recent work also shows that multimodal inputs can improve forecasting and make the signal more clinically aligned (Tan et al., 2024), while external validation studies show that model choice, preprocessing, and calibration matter at least as much as raw architecture novelty (Bashiri et al., 2024; Cao et al., 2022).

This matters for the proposed report because it shifts the novelty claim away from inventing yet another AKI predictor. A stronger claim is that high-quality AKI prediction already exists, but is not yet embedded in an agentic decision layer that can decide which model to use, what evidence is missing, and how to explain the result in a way that remains tied to the underlying quantitative substrate.

## Why drug-induced AKI is the best target use case

Drug-induced AKI is the best narrowing of the problem because it is both clinically grounded and operationally actionable. The DIKD standardization work gives a clearer phenotype and time-course frame for what should count as a drug-linked renal event (Mehta et al., 2015). ICU pharmaco-epidemiology also shows that nephrotoxic exposures are common, confounded, and worth monitoring with more care than simple exposure flags allow (Kom et al., 2023).

The prediction literature then adds a practical reason to focus here. Multicenter time-series modeling over nephrotoxic drugs is feasible (Heo et al., 2023), and adult nephrotoxin-focused prediction can substantially reduce false alerts relative to simpler exposure rules (Griffin et al., 2024). At the same time, the literature is split across drug-specific models such as vancomycin and cisplatin (Ambe et al., 2025; Mu et al., 2022; Okawa et al., 2022). That fragmentation is a weakness for a conventional one-model paper, but it is a strength for an agentic framework paper: a routing layer is easiest to justify when different subproblems already demand different models, variables, and timing assumptions.

## Research gap

The literature supports four narrow claims that together justify the gap.

- Quantitative AKI prediction is strong enough that the bottleneck has moved from raw prediction to model selection, integration, explanation, and deployment logic (Bashiri et al., 2024; Tan et al., 2024; Tomašev et al., 2019).
- Tool-using clinical LLMs can already retrieve and execute validated quantitative tools, but those systems have mostly been developed for generic risk calculators rather than longitudinal renal forecasting (Jin et al., 2025; Liu et al., 2025).
- Existing AKI-focused agent systems do not yet clearly demonstrate dynamic orchestration of external renal-risk models over evolving EHR trajectories (Gordon et al., 2025; T. Shi et al., 2024).
- Drug-induced AKI is clinically important, label-sensitive, and heterogeneous across drugs and patient subgroups, which makes it a strong setting for an agentic routing architecture rather than a single global predictor (Griffin et al., 2024; Heo et al., 2023; Mehta et al., 2015; Mu et al., 2022).

A defensible gap statement is therefore:

An agentic LLM framework for AKI should not be framed as a replacement for existing time-series predictors. It should be framed as a coordination layer that links structured EHR access, protocol-aware longitudinal reasoning, and external nephrotoxicity models in a way that current AKI predictors and current clinical LLM agents each only partially cover.

## Implications for a proposed framework

A plausible framework suggested by the literature has four parts.

| Component | Function | Why the literature supports it | Key papers |
|:---|:---|:---|:---|
| Quantitative substrate | Run one or more external AKI or nephrotoxicity predictors over structured EHR time series. | Strong predictive baselines already exist for ICU AKI and nephrotoxin-focused tasks. | (Griffin et al., 2024; Heo et al., 2023; Lyu et al., 2024; Tan et al., 2024; Tomašev et al., 2019) |
| Agentic routing layer | Decide which model, rule set, or drug-specific pathway is appropriate for the current patient. | Clinical agents can retrieve and execute validated tools, but have not been specialized to renal model routing. | (Jin et al., 2025; Liu et al., 2025) |
| EHR reasoning and safety layer | Pull missing variables, resolve time windows, apply KDIGO or DIKD logic, and maintain a compact longitudinal state. | EHR code agents and protocol-constrained reasoning already show the right primitives. | (Mehta et al., 2015; Qu & Farber, 2026; W. Shi et al., 2024) |
| Privacy layer | Use synthetic EHR mainly for pretraining, stress testing, or privacy-preserving development workflows. | DP sequence generators exist, but privacy should remain an infrastructure claim rather than the main scientific novelty. | (Lee et al., 2020; Torfi et al., 2020) |

The literature also suggests what the framework should avoid. A report-first multi-agent layer that only debates over static model outputs would be too close to existing work (Wang et al., 2024). A structure-following consensus layer without external quantitative model invocation would also fall short of the strongest novelty claim (Gordon et al., 2025).

## Evaluation plan

Because the report is meant to justify a research gap, the evaluation plan should test both the predictive substrate and the agentic layer.

| Evaluation question | What to measure | Why it matters | Literature basis |
|:---|:---|:---|:---|
| Does the full system predict AKI well enough to matter clinically? | AUROC, AUPRC, calibration, lead time, and alert burden such as false alerts per true alert. | AKI papers already make clear that clinical usefulness depends on more than discrimination alone. | (Bashiri et al., 2024; Tan et al., 2024; Tomašev et al., 2019; Yang et al., 2024) |
| Does the agent choose the right model or pathway? | Tool or model selection accuracy, parameter extraction accuracy, and success rate of executable calls. | This is the core novelty relative to standard AKI models. | (Jin et al., 2025; Liu et al., 2025; W. Shi et al., 2024) |
| Does broader AKI literature transfer cleanly to drug-induced AKI? | Performance across a generic AKI cohort and exposure-aware DI-AKI cohorts, with subgroup analyses by drug class. | Drug-induced AKI is heterogeneous and may require routing rather than one global model. | (Ambe et al., 2025; Griffin et al., 2024; Heo et al., 2023; Mu et al., 2022; Okawa et al., 2022) |
| Is the reasoning clinically auditable? | Agreement with KDIGO or DIKD logic, error rate in temporal rule application, and frequency of unsupported recommendations. | A safety case needs more than predictive accuracy. | (Lyu et al., 2024; Mehta et al., 2015; Qu & Farber, 2026) |
| Does the system generalize across sites? | External validation and recalibration across hospitals and subpopulations. | AKI models are sensitive to site shift and subgroup imbalance. | (Bashiri et al., 2024; Cao et al., 2022; Yang et al., 2024) |
| Does synthetic data help without becoming the main claim? | Utility gap between real-trained and synthetic-trained models, plus formal privacy budget reporting when DP is used. | Privacy is useful infrastructure, but weak synthetic fidelity would undermine the whole stack. | (Lee et al., 2020; Torfi et al., 2020) |

A good experimental hierarchy would therefore be:

1.  Compare the agentic system against strong non-agentic AKI baselines.
2.  Compare dynamic routing against a single fixed predictor.
3.  Test drug-specific cohorts where the value of routing should be largest.
4.  Test external transfer across sites.
5.  Test whether synthetic-data training or augmentation preserves enough utility to support the privacy claim.

## Bottom line

The most credible report line is not that LLMs will outperform existing AKI predictors on raw forecasting. The more credible line is that the literature has created the right pieces for a new systems contribution: strong longitudinal AKI models, early examples of tool-using clinical agents, and enough drug-induced AKI heterogeneity to justify model routing. The research gap is the missing integration layer that turns those pieces into an auditable agentic framework for nephrotoxicity prediction, with privacy-preserving synthetic EHR data serving as development infrastructure rather than the main contribution (Heo et al., 2023; Jin et al., 2025; Lee et al., 2020; Liu et al., 2025; Tomašev et al., 2019; Torfi et al., 2020).

---

## References

Ambe, K., Aoki, Y., Murashima, M., Wachino, C., Deki, Y., Ieda, M., Kondo, M., Furukawa-Hibi, Y., Kimura, K., Hamano, T., & Tohkin, M. (2025). Prediction of Cisplatin‐Induced Acute Kidney Injury Using an Interpretable Machine Learning Model and Electronic Medical Record Information. *Clinical and Translational Science*, *18*. <https://doi.org/10.1111/cts.70115>

Bashiri, F. S., Carey, K., Martin, J., Koyner, J., Edelson, D., Gilbert, E., Mayampurath, A., Afshar, M., & Churpek, M. (2024). Development and external validation of deep learning clinical prediction models using variable-length time series data. *Journal of the American Medical Informatics Association : JAMIA*. <https://doi.org/10.1093/jamia/ocae088>

Cao, J., Zhang, X., Shahinian, V., Yin, H., Steffick, D., Saran, R., Crowley, S., Mathis, M., Nadkarni, G., Heung, M., & Singh, K. (2022). Generalizability of an acute kidney injury prediction model across health systems. *Nature Machine Intelligence*, *4*, 1121–1129. <https://doi.org/10.1038/s42256-022-00563-8>

Gordon, D., Petousis, P., Nicholas, S. B., & Bui, A. A. T. (2025). AKIBoards: A Structure-Following Multiagent System for Predicting Acute Kidney Injury. *ArXiv*, *abs/2504.20368*. <https://doi.org/10.48550/arXiv.2504.20368>

Griffin, B. R., Mudireddy, A., Horne, B., Chonchol, M., Goldstein, S., Goto, M., Matheny, M. E., Street, W. N., Vaughan-Sarrazin, M., Jalal, D. I., & Misurac, J. (2024). Predicting Nephrotoxic Acute Kidney Injury in Hospitalized Adults: A Machine Learning Algorithm. *Kidney Medicine*, *6*. <https://doi.org/10.1016/j.xkme.2024.100918>

Heo, S., Kang, E., Yu, J., Kim, H., Lee, S., Kim, K., Hwangbo, Y., Park, R. W., Shin, H., Ryu, K., Kim, C., Jung, H., Chegal, Y., Lee, J.-H., & Park, Y. (2023). Time Series AI Model for Acute Kidney Injury Detection Based on a Multicenter Distributed Research Network: Development and Verification Study. *JMIR Medical Informatics*, *12*. <https://doi.org/10.2196/47693>

Jin, Q., Wang, Z., Yang, Y., Zhu, Q., Wright, D., Huang, T., Khandekar, N., Wan, N., Ai, X., Wilbur, W., He, Z., Taylor, R. A., Chen, Q., & Lu, Z. (2025). AgentMD: Empowering language agents for risk prediction with large-scale clinical tool learning. *Nature Communications*, *16*. <https://doi.org/10.1038/s41467-025-64430-x>

Kom, I. A. R. Y., Dongelmans, D., Abu-Hanna, A., Schut, M., Lange, D. D. de, Roon, E. V. van, Jonge, E. de, Bouman, C., Keizer, N. D. de, Jager, K., & Klopotowska, J. E. (2023). Acute kidney injury associated with nephrotoxic drugs in critically ill patients: a multicenter cohort study using electronic health record data. *Clinical Kidney Journal*, *16*, 2549–2558. <https://doi.org/10.1093/ckj/sfad160>

Lee, D., Yu, H., Jiang, X., Rogith, D., Gudala, M., Tejani, M., Zhang, Q., & Xiong, L. (2020). Generating sequential electronic health records using dual adversarial autoencoder. *Journal of the American Medical Informatics Association : JAMIA*, *27 9*, 1411–1419. <https://doi.org/10.1093/jamia/ocaa119>

Liu, F., Wu, J., Zhou, H., Gu, X., Molaei, S., Thakur, A., Clifton, L., Wu, H., & Clifton, D. A. (2025). RiskAgent: Autonomous Medical AI Copilot for Generalist Risk Prediction. *ArXiv*, *abs/2503.03802*. <https://doi.org/10.1101/2025.04.03.25323489>

Lyu, X., Fan, B., Hüser, M., Hartout, P., Gumbsch, T., Faltys, M., Merz, T., Rätsch, G., & Borgwardt, K. M. (2024). An empirical study on KDIGO-defined acute kidney injury prediction in the intensive care unit. *Bioinformatics*, *40*, i247–i256. <https://doi.org/10.1093/bioinformatics/btae212>

Mehta, R., Awdishu, L., Davenport, A., Murray, P., Macedo, E., Cerdá, J., Chakaravarthi, R., Holden, A. L., & Goldstein, S. (2015). Phenotype Standardization for Drug Induced Kidney Disease. *Kidney International*, *88*, 226–234. <https://doi.org/10.1038/ki.2015.115>

Mu, F., Cui, C., Tang, M., Guo, G., Zhang, H., Ge, J., Bai, Y., Zhao, J., Cao, S., Wang, J., & Guan, Y. (2022). Analysis of a machine learning–based risk stratification scheme for acute kidney injury in vancomycin. *Frontiers in Pharmacology*, *13*. <https://doi.org/10.3389/fphar.2022.1027230>

Okawa, T., Mizuno, T., Hanabusa, S., Ikeda, T., Mizokami, F., Koseki, T., Takahashi, K., Yuzawa, Y., Tsuboi, N., Yamada, S., & Kameya, Y. (2022). Prediction model of acute kidney injury induced by cisplatin in older adults using a machine learning algorithm. *PLoS ONE*, *17*. <https://doi.org/10.1371/journal.pone.0262021>

Qu, Z., & Farber, M. K. (2026). *Vital Trace: Protocol-Constrained Patient-State Reasoning for Longitudinal Clinical Trajectories*.

Shi, T., Xiao, M., Xu, H., Zhao, H., & Kong, G. (2024). AKI-Detector: A Multi-Agent Framework by Integrating Machine Learning and Large Language Models for Early Prediction of Acute Kidney Injury in ICU. *AMIA ... Annual Symposium Proceedings. AMIA Symposium*, *2024*, 1190–1199.

Shi, W., Xu, R., Zhuang, Y., Yu, Y., Zhang, J., Wu, H., Zhu, Y., Ho, J. C., Yang, C., & Wang, M. D. (2024). EHRAgent: Code Empowers Large Language Models for Few-shot Complex Tabular Reasoning on Electronic Health Records. *Proceedings of the Conference on Empirical Methods in Natural Language Processing. Conference on Empirical Methods in Natural Language Processing*, *2024*, 22315–22339. <https://doi.org/10.18653/v1/2024.emnlp-main.1245>

Tan, Y., Dede, M., Mohanty, V., Dou, J., Hill, H., Bernstam, E. V., & Chen, K. (2024). Forecasting acute kidney injury and resource utilization in ICU patients using longitudinal, multimodal models. *Journal of Biomedical Informatics*, 104648. <https://doi.org/10.1016/j.jbi.2024.104648>

Tomašev, N., Glorot, X., Rae, J. W., Zielinski, M., Askham, H., Saraiva, A., Mottram, A., Meyer, C., Ravuri, S. V., Protsyuk, I. V., Connell, A., Hughes, C. O., Karthikesalingam, A., Cornebise, J., Montgomery, H., Rees, G., Laing, C., Baker, C., Peterson, K. S., … Mohamed, S. (2019). A Clinically Applicable Approach to Continuous Prediction of Future Acute Kidney Injury. *Nature*, *572*, 116–119. <https://doi.org/10.1038/s41586-019-1390-1>

Torfi, A., Fox, E., & Reddy, C. K. (2020). Differentially Private Synthetic Medical Data Generation using Convolutional GANs. *Inf. Sci.*, *586*, 485–500. <https://doi.org/10.1016/j.ins.2021.12.018>

Wang, Z., Zhu, Y., Zhao, H., Zheng, X., Wang, T., Tang, W., Wang, Y., Pan, C., Harrison, E. M., Gao, J., & Ma, L. (2024). ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration. *Proceedings of the ACM on Web Conference 2025*. <https://doi.org/10.1145/3696410.3714877>

Yang, M., Liu, S., Hao, T., Ma, C., Chen, H., Li, Y., Wu, C., Xie, J., Qiu, H., Li, J., Yang, Y., & Liu, C. (2024). Development and validation of a deep interpretable network for continuous acute kidney injury prediction in critically ill patients. *Artificial Intelligence in Medicine*, *149*, 102785. <https://doi.org/10.1016/j.artmed.2024.102785>
