import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==============================================================================
# VAST.AI PRODUCTION ATTENTION EXTRACTION PIPELINE
# ==============================================================================
# Prerequisites on the Vast instance:
# pip install torch transformers peft accelerate matplotlib seaborn
# ==============================================================================

def process_pipeline():
    # 1. Configuration
    base_model_name = "google/gemma-4-12b-it"
    adapter_name = "QinEmPeRoR93/paper-gemma-4-12b-aki-lora"
    input_file = "data/eval_holdout.jsonl"
    out_dir = "plots/attention_heatmaps"
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Load Model and Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print("Loading base model in half precision (float16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        device_map="auto", 
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    
    print(f"Fusing PEFT adapter: {adapter_name}...")
    model = PeftModel.from_pretrained(base_model, adapter_name)
    model.eval()

    # 3. Read Evaluation Data
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing input dataset: {input_file}")
        
    print(f"Looping through {input_file} to generate batch heatmaps...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            messages = data.get("messages", [])
            if not messages or len(messages) < 3:
                continue
                
            print(f"Processing Patient {idx+1}...")
            
            # Format using tokenizer chat template (including the assistant's final answer)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"][0]
            tokens = [tokenizer.decode([tid]) for tid in input_ids]
            
            # Run forward pass
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True
                )
            
            # Extract final layer attention weights
            attention_matrix = outputs.attentions[-1][0] # Get final layer, remove batch
            
            # Use max across heads to capture specific head focus (prevents dilution from sink heads)
            max_attention = attention_matrix.max(dim=0)[0].cpu().float().numpy() # Shape: [seq, seq]
            
            # Find the index of the classification token ([AKI_STAGE_1+] or [NORMAL])
            target_idx = None
            for target_word in ["STAGE", "NORMAL", "IMMINENT"]:
                indices = [i for i, t in enumerate(tokens) if target_word in t]
                if indices:
                    target_idx = indices[-1]
                    break
            
            if target_idx is None:
                # Fallback to the last non-special token if classification label isn't found
                target_idx = len(tokens) - 3
                
            # Get the attention weights from our target classification token to all previous tokens
            target_attention = max_attention[target_idx, :target_idx+1]
            
            # Zero out self-attention and bos token attention to prevent them from washing out clinical features
            target_attention[target_idx] = 0.0
            target_attention[0] = 0.0
            
            # Filter tokens: we only want to keep meaningful clinical tokens, 
            # and ignore system prompt headers, punctuation, newlines, and template tags.
            filtered_indices = []
            stop_words = {
                "<", ">", "start_of_turn", "model", "user", "end_of_turn", "turn", "\n", " ", ":", ".", "[", "]", "|", ",", "-",
                "You", "are", "an", "AI", "enabled", "AI-enabled", "system", "clinical", "safety", "sentinel", "Your", "task", "is", "to", "continuous", "monitor", 
                "ICU", "patient", "trajectories", "and", "predict", "the", "imminent", "onset", "of", "Medication-Induced", "Kidney", 
                "Injury", "demographics", "yo", "Sex", "Baseline", "Serum", "Creatinine", "Initiating", "sequence", "assessed", "currently",
                "status", "risk", "indicates", "combined", "exposure", "cumulative", "received", "nephrotoxic", "antibiotics", "hemodynamic", 
                "parameters", "active", "Meds", "Active", "as", "Synthesis"
            }
            
            for i in range(target_idx + 1):
                tok_str = tokens[i].strip()
                # Case-insensitive boilerplate checks
                if tok_str and not any(s.lower() in tok_str.lower() for s in stop_words):
                    filtered_indices.append(i)
            
            # Get the top 20 most attended tokens from our filtered list
            sorted_by_attn = sorted(filtered_indices, key=lambda x: target_attention[x], reverse=True)
            top_n_indices = sorted_by_attn[:20]
            
            # Sort the selected indices chronologically so they read left-to-right in order of the prompt
            top_n_indices.sort()
            
            attention_slice = target_attention[top_n_indices]
            # Re-normalize over just the selected clinical tokens
            sum_slice = attention_slice.sum()
            if sum_slice > 0:
                attention_slice = attention_slice / sum_slice
            attention_slice = attention_slice.reshape(1, -1)
            
            slice_tokens = [tokens[i].strip() for i in top_n_indices]

            # 4. Generate Heatmap
            plt.figure(figsize=(12, 4))
            sns.heatmap(
                attention_slice, 
                xticklabels=slice_tokens, 
                yticklabels=[f"Focus of '{tokens[target_idx].strip()}'"],
                cmap="Oranges", 
                cbar_kws={'label': 'Max Attention Weight'}, 
                square=True, 
                linewidths=0.5, 
                linecolor='gray',
                annot=True,
                fmt=".2f"
            )
            
            plt.title(f"Patient {idx+1} Causal Attention Mapping (Label: {tokens[target_idx].strip()})", fontsize=12, pad=15)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=10, fontweight='bold')
            
            out_path = os.path.join(out_dir, f"patient_{idx+1}_attention_heatmap.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            
    print(f"Pipeline complete! Saved all patient heatmaps to {out_dir}/")

if __name__ == "__main__":
    process_pipeline()
