import os
import json
import numpy as np
import torch
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model

def get_peft_timesfm():
    print("Loading TimesFM 2.5 Base Model...")
    import timesfm
    torch.set_float32_matmul_precision("high")
    
    # Load TimesFM 2.5 Base Model from Hugging Face
    model_path = "google/timesfm-2.5-200m-pytorch"
    base_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path)
    
    # Configure LoRA to target the Transformer attention mechanism
    # TimesFM 2.5 uses standard attention projections, we target them to inject knowledge
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["qkv_proj", "out", "ff0", "ff1"],
        lora_dropout=0.05,
        bias="none",
    )
    
    print("Applying LoRA adapters for Parameter-Efficient Fine-Tuning (PEFT)...")
    # Access the underlying PyTorch nn.Module (in TimesFM this is typically .model)
    pytorch_module = getattr(base_model, 'model', base_model)
    
    try:
        peft_model = get_peft_model(pytorch_module, lora_config)
        peft_model.print_trainable_parameters()
    except ValueError as e:
        print(f"Note: Standard HuggingFace target modules not perfectly aligned. {e}")
        # Fallback config for generic modules if names differ in custom architecture
        lora_config.target_modules = ["query", "value"] 
        peft_model = get_peft_model(pytorch_module, lora_config)
        peft_model.print_trainable_parameters()
    print("LoRA weights loaded successfully.")
    return base_model, peft_model

def compute_timesfm_patch_loss(peft_model, context, target, covs=None):
    """
    Computes the true mathematical Mean Squared Error (MSE) by tracing the 
    PyTorch Autograd graph through the TimesFM LoRA adapters.
    """
    import torch
    import torch.nn.functional as F
    
    # TimesFM's internal transformer expects patched inputs of size 63 (plus 1 for mask)
    # We pad our context and covariates to fill the 64-length patch dimension
    padded_input = np.zeros(63, dtype=np.float32)
    
    # Pack context (SCr days 1-3)
    padded_input[:len(context)] = context
    
    if covs is not None:
        # Pack covariates (MAP, Vanco, Zosyn) into the sequence patch
        padded_input[3:6] = covs[:3, 0]
        padded_input[6:9] = covs[:3, 1]
        padded_input[9:12] = covs[:3, 2]
        
    device = next(peft_model.parameters()).device
    inputs = torch.tensor(padded_input, device=device).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, 63]
    masks = torch.ones(1, 1, 1, dtype=torch.float32, device=device)
    
    # 1. Forward Pass (Tracks gradients through LoRA adapters)
    # The peft_model wraps TimesFM's internal nn.Module
    out = peft_model(inputs, masks)
    
    # 2. Extract Point Forecast
    # TimesFM returns a nested tuple; out[0][2] is 'output_ts' which passes through LoRA
    preds = out[0][2].squeeze().flatten()
    
    # 3. Map to Target Horizon (2 days)
    # We map the latent output projection back to our 2-day ground truth horizon
    preds_mapped = preds[:2]
    true_target = torch.tensor(target, dtype=torch.float32, device=device)
    
    # 4. Compute True Mathematical Loss (MSE)
    loss = F.mse_loss(preds_mapped, true_target)
    return loss

def main():
    data_path = "output/timesfm_training_cohort.jsonl"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    patients = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            patients.append(json.loads(line))
            
    print(f"Loaded {len(patients)} patients for Fine-Tuning.")
    
    # 1. Initialize PEFT Model
    base_model, peft_model = get_peft_timesfm()
    
    # 2. Setup Optimizer
    optimizer = AdamW(peft_model.parameters(), lr=1e-4)
    epochs = 5
    
    print(f"\nStarting LoRA Fine-Tuning for {epochs} epochs...")
    peft_model.train()
    
    # 3. Training Loop
    # Note: In a full TimesFM production pipeline, data is formatted into patched tensors.
    # This loop demonstrates the PEFT pipeline structure optimizing the temporal horizon.
    peft_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for p in patients:
            # Extract SCr timeline and Covariates (Vancomycin, Zosyn, MAP)
            scr = np.array(p['scr'], dtype=np.float32)
            
            # Extract dynamic covariates
            map_series = np.array(p['map'], dtype=np.float32)
            vanco_series = np.array(p['vanco_trough'], dtype=np.float32)
            zosyn_series = np.array([1.0 if x else 0.0 for x in p['zosyn_active']], dtype=np.float32)
            
            # Stack into multivariate context tensor (Shape: [5 days, 3 features])
            dynamic_covariates = np.stack([map_series, vanco_series, zosyn_series], axis=-1)
            
            # Compute True PyTorch Loss (MSE) by routing inputs through the LoRA graph
            loss = compute_timesfm_patch_loss(peft_model, context=scr[:3], target=scr[3:], covs=dynamic_covariates)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(patients)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_loss:.4f}")
        
    print("\nFine-tuning complete!")
    
    # 4. Save the Adapter Weights
    lora_output_dir = "output/lora_weights"
    os.makedirs(lora_output_dir, exist_ok=True)
    peft_model.save_pretrained(lora_output_dir)
    print(f"Saved optimized LoRA adapters to {lora_output_dir}/")
    print("To run inference, load these weights via `peft_model.from_pretrained()` and evaluate!")

if __name__ == "__main__":
    main()
