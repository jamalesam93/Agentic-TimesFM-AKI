# EHR Synthesis Engine: Vancomycin-Zosyn Nephrotoxicity Sentinel

A modular, privacy-preserving clinical simulation engine that extracts statistical parameters from raw ICU EHR data and synthesizes realistic longitudinal patient trajectories. The engine is optimized to generate fine-tuning payloads for AI-based clinical safety sentinels monitoring drug-induced Acute Kidney Injury (AKI).

---

## 📂 Project Structure

```text
AKI/
├── src/
│   ├── __init__.py          # Marks src as a Python package
│   ├── data_extraction.py   # Historical EHR simulation & DP/parametric parameter extraction
│   ├── generator.py         # Cohort baseline synthesis & temporal trajectory simulation
│   └── textualization.py   # Serializers for LLM JSONL and clinical markdown formats
├── output/                  # Default target directory for pipeline outputs
│   ├── reports/             # Individual clinical markdown patient charts
│   ├── raw_historical_cohort.csv
│   ├── extracted_parameters.json
│   ├── synthetic_cohort_baselines.csv
│   └── llm_fine_tuning_dataset.jsonl
├── main.py                  # CLI Orchestrator and pipeline runner
├── requirements.txt         # Package dependencies (including diffprivlib)
└── README.md                # System documentation & clinical background
```

---

## ⚕️ Clinical Significance: Vancomycin-Zosyn Synergistic Nephrotoxicity

### The Therapeutic Dilemma
In intensive care units (ICUs) and emergency departments, the combination of **Vancomycin** (a glycopeptide antibiotic targeting Gram-positive pathogens like MRSA) and **Piperacillin-Tazobactam** (known brand name **Zosyn**, a beta-lactam/beta-lactamase inhibitor targeting Gram-negative pathogens including *Pseudomonas aeruginosa*) is one of the most frequently prescribed empirical regimens for patients presenting with severe sepsis, septic shock, or healthcare-associated pneumonia.

While providing broad-spectrum coverage, co-administration of these two agents is associated with a **highly elevated risk of synergistic acute kidney injury (AKI)**.

### Pathophysiological and Epidemiological Mechanism
1. **Vancomycin Monotherapy Nephrotoxicity**: Vancomycin is known to accumulate in renal proximal tubule cells, triggering oxidative stress, mitochondrial dysfunction, cellular necrosis, and allergic interstitial nephritis. Risk is dose-dependent and managed via Therapeutic Drug Monitoring (TDM) targeting specific troughs (15–20 µg/mL) or AUC/MIC ratios.
2. **Piperacillin-Tazobactam Monotherapy**: Zosyn is rarely direct-nephrotoxic on its own.
3. **The Synergistic Effect**: When administered together, epidemiological studies consistently report AKI rates of **15% to 35%**, compared to **8% to 15%** for Vancomycin alone. The exact mechanism remains debated, but evidence suggests:
   - **Competed Secretion / Accumulation**: Piperacillin may competitively inhibit active organic anion transporters (OAT) in renal tubules, reducing clearance rates of Vancomycin, thus leading to toxic intracellular accumulation.
   - **Subclinical Damage Exacerbation**: Zosyn may exacerbate mild cellular stress induced by Vancomycin, pushing borderline tubular cells into acute tubular necrosis (ATN).
4. **Clinical Implications**: This synergistic injury is characterized by a rapid onset (typically within 3–5 days of co-administration) and is measured clinically by an increase in Serum Creatinine (SCr) and a drop in urine output. An AI sentinel that can detect early physiologic markers (e.g., subtle changes in hemodynamics like Mean Arterial Pressure combined with cumulative exposure) is critical for early warning.

---

## 🔄 System Flow & Architecture

The pipeline consists of five stages designed to preserve privacy while maintaining clinical correlation:

1. **Stage 0: Raw Historical Data Simulation**: Simulates historical patient records (representing raw data sources like MIMIC-IV). This contains demographics, drug exposure logs (Vancomycin, Zosyn), baseline lab results (Serum Creatinine), and outcome status (developed AKI).
2. **Stage 1: Parametric & DP Parameter Extraction**: Computes demographic marginals, extracts the mean and variance of log-transformed baseline Serum Creatinine, and maps age/gender correlations in log-space. If `--epsilon` is enabled, applies differential privacy via Laplace noise perturbation to ensure individual patient data cannot be reconstructed from the statistics.
3. **Stage 2: Baseline Cohort Synthesis**: Reads the extracted parameters JSON and samples clean baseline values (age, gender, baseline SCr, antibiotic prescriptions, and AKI risk). No historical records are ever checked during this phase.
4. **Stage 3: Longitudinal ICU Trajectory Simulation**: Generates a 5-day longitudinal timeseries. It models daily hemodynamics (Mean Arterial Pressure), cumulative Vancomycin drug troughs (which accumulate faster and higher under the synergistic influence of Zosyn), and models Serum Creatinine rise for patients flagged with AKI.
5. **Stage 4: Textualization & LLM Serialization**: Renders two outputs:
   - **LLM Fine-Tuning JSONL**: Prompt-response pairs formatted for conversational fine-tuning of clinical sentinel AI models.
   - **Clinical Markdown Reports**: Detailed, print-ready clinician charts showing the chronological flowsheet, patient demographics, and automated drug-interaction assessment.

---

## 🚀 Execution & CLI Guide

### 1. Prerequisites & Installation
Ensure you have Python 3.8+ installed. Clone or copy the workspace files and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Basic Pipeline Run
Generate a default cohort (simulating 500 historical patients, synthesizing 10 clean patients, and saving 5 clinical markdown reports) in the default `output/` directory:

```bash
python main.py
```

### 3. CLI Arguments Configuration
The `main.py` orchestrator supports full customization:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--n-patients` | `int` | `500` | Size of raw historical database to simulate |
| `--n-synthetic` | `int` | `10` | Size of the synthetic patient cohort to generate |
| `--days` | `int` | `5` | Length of ICU simulation timeline (days) |
| `--epsilon` | `float` | `None` | Privacy budget ($\epsilon$) for Differential Privacy. Omit to disable. |
| `--output-dir` | `str` | `"output"`| Output directory path |
| `--seed` | `int` | `42` | Random seed for reproducibility |
| `--save-reports`| `int` | `5` | Number of detailed markdown clinical reports to export |

### 4. Advanced Run Examples

#### Running with Differential Privacy
To run the extraction pipeline under formal $( \epsilon, 0 )$-differential privacy guarantees:
```bash
python main.py --n-patients 1000 --n-synthetic 50 --epsilon 1.0 --output-dir dp_output
```

#### Large-Scale Dataset Generation
Generate 1,000 synthetic patient sequences for LLM fine-tuning without saving excessive Markdown files:
```bash
python main.py --n-patients 2000 --n-synthetic 1000 --save-reports 0 --output-dir model_dataset
```

---

## 📊 Verification of Outputs
The generated files will populate your chosen output directory:
- `raw_historical_cohort.csv`: Clean baseline of the initial "PhysioNet" simulation.
- `extracted_parameters.json`: The statistical parameters used for baseline sampling.
- `llm_fine_tuning_dataset.jsonl`: Ready-to-use JSON Lines training payload.
- `reports/patient_SYN_XXXXX_report.md`: Visual, human-readable patient charts.
