import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# VAST.AI / NATIVE GPU ATTENTION EXTRACTION PIPELINE
# ==============================================================================
# PREREQUISITES for the Vast.ai instance:
# pip install torch transformers matplotlib seaborn accelerate
# ==============================================================================

def main():
    print("Initializing Native Model Loading...")
    
    # 1. Load the RAW HuggingFace Model (Not GGUF)
    # On your Vast.ai instance, replace this path with the path to your raw safetensors
    # You will likely want to load in 16-bit or 8-bit precision to save VRAM on an A100
    model_id = "google/gemma-4-12b" # (Or your local fine-tuned directory)
    
    print(f"Loading {model_id} into VRAM. This will take substantial memory...")
    # NOTE: We use device_map="auto" to automatically spread across available GPUs
    # torch_dtype=torch.float16 ensures we don't blow up the VRAM.
    
    # from peft import PeftModel
    # 
    # base_model_id = "google/gemma-4-12b"
    # adapter_id = "QinEmPeRoR93/phd-gemma-4-12b-aki-lora"
    # 
    # tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_id, 
    #     device_map="auto", 
    #     torch_dtype=torch.float16
    # )
    # model = PeftModel.from_pretrained(base_model, adapter_id)
    
    print("[SIMULATION MODE] Running heatmap generation logic using mock attention vectors...")
    print("This script is designed to be executed on your future Vast.ai instance where you will uncomment the PEFT model loading code.")
    
    # 2. Define a Patient Trajectory
    patient_text = (
        "You are an AI-enabled clinical safety sentinel.\n"
        "Patient: 55 yo, Sex: M. Baseline SCr: 1.68 mg/dL.\n"
        "[Day 3] MAP: 62 mmHg | Vanco Trough: 26.2 ug/mL | SCr: 3.52 mg/dL\n"
        "Prediction: [AKI_STAGE_1+]"
    )
    
    # In a real environment, you encode the text:
    # inputs = tokenizer(patient_text, return_tensors="pt").to("cuda")
    # input_ids = inputs["input_ids"]
    # tokens = [tokenizer.decode([i]) for i in input_ids[0]]
    
    # For demonstration, we will manually tokenize the text to show the heatmap logic:
    tokens = [
        "Patient", ":", " 55", " yo,", " Baseline", " SCr", ":", " 1.68", 
        " MAP", ":", " 62", " Vanco", " Trough", ":", " 26.2", " SCr", ":", " 3.52",
        " Prediction", ":", " [", "AKI", "_STAGE", "_1", "+]"
    ]
    
    # 3. Capture Attention (THE MAGIC HAPPENS HERE)
    # outputs = model.forward(
    #     **inputs, 
    #     output_attentions=True,  # Crucial flag: forces PyTorch to return internal weights
    #     return_dict=True
    # )
    
    # The 'outputs.attentions' object is a tuple of tensors.
    # Dimensions: (num_layers, batch_size, num_heads, sequence_length, sequence_length)
    # Example: (40 layers, 1, 16 heads, 25 tokens, 25 tokens)
    
    print("Extracting attention weights from the final transformer layer...")
    
    # 4. Aggregate Attention Weights
    # We want to know: "When predicting 'AKI', what previous words did the model look at?"
    # We take the final layer's attention matrix, and average the weights across all attention heads.
    # attention_matrix = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()
    
    # MOCKING THE ATTENTION MATRIX FOR DEMONSTRATION
    # We will simulate the model attending heavily to "Vanco", "26.2", "SCr", and "3.52"
    seq_len = len(tokens)
    mock_attention = np.zeros((seq_len, seq_len))
    
    # Models only look backward (causal masking). 
    # Let's define the attention weights for the 'AKI' token (Index 21)
    aki_idx = tokens.index("AKI")
    
    for i in range(aki_idx + 1):
        # Baseline noise
        mock_attention[aki_idx, i] = np.random.uniform(0.01, 0.05)
        
    # Inject high attention spikes where the model is "focusing" its clinical logic
    mock_attention[aki_idx, tokens.index(" Vanco")] = 0.85
    mock_attention[aki_idx, tokens.index(" 26.2")] = 0.92
    mock_attention[aki_idx, tokens.index(" MAP")] = 0.60
    mock_attention[aki_idx, tokens.index(" 62")] = 0.70
    mock_attention[aki_idx, tokens.index(" SCr")] = 0.40
    mock_attention[aki_idx, tokens.index(" 3.52")] = 0.95
    mock_attention[aki_idx, tokens.index(" Baseline")] = 0.30
    mock_attention[aki_idx, tokens.index(" 1.68")] = 0.50
    
    # Normalize the row so it sums to 1.0 (softmax property of attention)
    mock_attention[aki_idx, :] /= mock_attention[aki_idx, :].sum()

    # 5. Visualize the Heatmap
    print("Generating attention heatmap visualization...")
    plt.figure(figsize=(14, 8))
    
    # We will plot just a specific slice: How much every token attends to previous tokens.
    # For clarity in the diagram, let's just plot the attention weights OF the "AKI" token
    # directed towards the input context.
    
    attention_slice = mock_attention[aki_idx, :aki_idx+1].reshape(1, -1)
    
    sns.heatmap(
        attention_slice,
        xticklabels=tokens[:aki_idx+1],
        yticklabels=["Attention from 'AKI'"],
        cmap="YlOrRd",  # Yellow to Red colormap (Red = High Attention)
        cbar_kws={'label': 'Attention Weight'},
        square=True,
        linewidths=1,
        linecolor='black'
    )
    
    plt.title("LLM Internal Attention Weights: What did the model 'see' when predicting AKI?", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "simulated_attention_heatmap.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Success! Attention heatmap saved to {out_path}")

if __name__ == "__main__":
    main()
