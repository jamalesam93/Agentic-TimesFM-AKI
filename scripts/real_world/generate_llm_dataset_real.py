import os
import sys
import json
import random

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import data_extraction_real as ehr
from src import generator
from src.textualization import format_to_llm_jsonl

def generate_dataset(n_patients=2000, seed=42):
    random.seed(seed)
    print("1. Extracting parameters from real historical EHR data...")
    raw_cohort = ehr.load_real_historical_data(n_patients=1000, seed=seed)
    extracted_parameters = ehr.extract_statistical_parameters(raw_cohort)
    
    print(f"2. Synthesizing {n_patients} privacy-preserving longitudinal patients...")
    synthetic_base = generator.synthesize_cohort(extracted_parameters, n_synthetic=n_patients, seed=seed)
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, "data", "real_world")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"3. Generating temporal trajectories and writing LLM training data to {data_dir}...")
    result = generator.process_cohort_parallel(
        patients=synthetic_base,
        output_dir=data_dir,
        days=5,
        base_seed=seed,
        max_workers=os.cpu_count(),
        save_reports=0,
        show_progress=True,
    )
    
    # Rename default names to the PhD expected names
    old_llm = os.path.join(data_dir, "llm_fine_tuning_dataset.jsonl")
    new_llm = os.path.join(data_dir, "paper_sft_dataset.jsonl")
    if os.path.exists(old_llm):
        if os.path.exists(new_llm):
            os.remove(new_llm)
        os.rename(old_llm, new_llm)
        
    old_timesfm = os.path.join(data_dir, "timesfm_training_cohort.jsonl")
    new_timesfm = os.path.join(data_dir, "paper_timesfm_dataset.jsonl")
    if os.path.exists(old_timesfm):
        if os.path.exists(new_timesfm):
            os.remove(new_timesfm)
        os.rename(old_timesfm, new_timesfm)

    print("\nDataset generation complete! Ready for Phase 2 LLM Fine-Tuning.")

if __name__ == "__main__":
    generate_dataset()
