import os
import sys
import json
import random

# Add parent directory to path so we can import ehr_synthesis_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ehr_synthesis_engine as ehr

def generate_dataset(n_patients=2000, seed=42):
    random.seed(seed)
    print("1. Extracting parameters from mock historical EHR data...")
    raw_cohort = ehr.generate_mock_historical_data(n_patients=1000, seed=seed)
    extracted_parameters = ehr.extract_statistical_parameters(raw_cohort)
    
    print(f"2. Synthesizing {n_patients} privacy-preserving longitudinal patients...")
    synthetic_base = ehr.synthesize_cohort(extracted_parameters, n_synthetic=n_patients, seed=seed)
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    jsonl_path = os.path.join(data_dir, "phd_proposal_sft_dataset.jsonl")
    
    print(f"3. Generating temporal trajectories and writing LLM training data to {jsonl_path}...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, patient in enumerate(synthetic_base):
            trajectory = ehr.generate_temporal_record(patient, seed=seed + i)
            llm_sample = ehr.format_to_llm_jsonl(patient, trajectory)
            f.write(json.dumps(llm_sample) + "\n")
            
            if (i + 1) % 500 == 0:
                print(f"   -> Processed {i + 1} / {n_patients} trajectories")

    print("\nDataset generation complete! Ready for Phase 2 LLM Fine-Tuning.")

if __name__ == "__main__":
    generate_dataset()
