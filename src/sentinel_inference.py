import os
import json
import numpy as np
import pandas as pd
import argparse

# Lazy loading of heavy models
def get_timesfm_model():
    print("Loading TimesFM 2.5 model...")
    import torch
    import timesfm
    from peft import PeftModel
    
    torch.set_float32_matmul_precision("high")
    
    import os
    # Initialize the pre-trained TimesFM model from Hugging Face directly
    model_path = "google/timesfm-2.5-200m-pytorch"
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path)
    
    # Attach LoRA weights if they exist
    lora_path = "exports/real_world/timesfm_lora"
    if os.path.exists(lora_path):
        print(f"Attaching specialized LoRA weights from {lora_path}...")
        pytorch_module = getattr(model, 'model', model)
        pytorch_module = PeftModel.from_pretrained(pytorch_module, lora_path)
        if hasattr(model, 'model'):
            model.model = pytorch_module
    else:
        print("No LoRA weights found. Running in Zero-Shot mode.")
        
    # Compile with forecasting configurations
    model.compile(
        timesfm.ForecastConfig(
            max_context=16, # Small context since we only have 3 days
            max_horizon=2,  # Predict next 2 days
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            return_backcast=True, # Required for covariate forecasting in 2.5
        )
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot TimesFM Inference for AKI Sentinel")
    parser.add_argument("--data-path", type=str, default="data/real_world/phd_proposal_timesfm_dataset.jsonl", help="Path to TimesFM JSONL dataset")
    parser.add_argument("--limit", type=int, default=100, help="Number of patients to evaluate")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Dataset not found at {args.data_path}")
        return

    # Load dataset
    patients = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            patients.append(json.loads(line))
            if len(patients) >= args.limit:
                break
    
    print(f"Loaded {len(patients)} patient trajectories for inference.")
    
    # Prepare inputs
    # We will use Day 1, 2, 3 as context and Day 4, 5 as ground truth
    context_inputs = []
    ground_truths = []
    synthetic_ids = []
    
    # Covariate lists (must span full 5 days: context + horizon)
    map_covs = []
    vanco_covs = []
    zosyn_covs = []
    
    for p in patients:
        scr_series = p['scr']
        if len(scr_series) >= 5:
            # Context: days 1-3
            context_inputs.append(np.array(scr_series[:3], dtype=np.float32))
            # Ground truth: days 4-5
            ground_truths.append(np.array(scr_series[3:5], dtype=np.float32))
            synthetic_ids.append(p['synthetic_id'])
            
            # Extract covariates
            map_covs.append(np.array(p['map'][:5], dtype=np.float32))
            vanco_covs.append(np.array(p['vanco_trough'][:5], dtype=np.float32))
            zosyn_covs.append(np.array([1.0 if x else 0.0 for x in p['zosyn_active'][:5]], dtype=np.float32))
            
    if not context_inputs:
        print("No valid patients found with 5 days of data.")
        return
        
    model = get_timesfm_model()
    
    print(f"Running forecasting with Multivariate Covariates on {len(context_inputs)} patients...")
    # Forecast returns a point forecast and a quantile forecast
    point_forecast, quantile_forecast = model.forecast_with_covariates(
        inputs=context_inputs,
        dynamic_numerical_covariates={
            "map": map_covs,
            "vanco_trough": vanco_covs,
            "zosyn_active": zosyn_covs
        }
    )
    
    # Calculate Mean Absolute Error (MAE)
    total_mae = 0.0
    total_patients_evaluated = len(context_inputs)
    
    print("\n--- Zero-Shot Inference Results ---")
    for i in range(total_patients_evaluated):
        pred = point_forecast[i]
        actual = ground_truths[i]
        
        mae = np.mean(np.abs(pred - actual))
        total_mae += mae
        
        # Display first few results
        if i < 5:
            print(f"Patient {synthetic_ids[i]}:")
            print(f"  Context SCr (Day 1-3): {context_inputs[i]}")
            print(f"  Actual SCr  (Day 4-5): {actual}")
            print(f"  Pred SCr    (Day 4-5): {pred}")
            print(f"  MAE: {mae:.3f}")
            print("-" * 30)
            
    avg_mae = total_mae / total_patients_evaluated
    print(f"\nOverall Mean Absolute Error (MAE) across {total_patients_evaluated} patients: {avg_mae:.3f} mg/dL")
    
    print("\nNext Steps: Consider fine-tuning the model using PEFT (LoRA) and covariates to improve predictive accuracy for the specific Vancomycin-Zosyn drug interaction.")

if __name__ == "__main__":
    main()
