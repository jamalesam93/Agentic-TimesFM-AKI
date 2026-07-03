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
    adapter_name = "QinEmPeRoR93/phd-gemma-4-12b-aki-lora"
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
        torch_dtype=torch.float16
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
            if not messages or len(messages) < 2:
                continue
                
            print(f"Processing Patient {idx+1}...")
            
            # Format using tokenizer chat template (system + user messages)
            inference_messages = [
                {"role": messages[0]["role"], "content": messages[0]["content"]},
                {"role": messages[1]["role"], "content": messages[1]["content"]},
            ]
            prompt = tokenizer.apply_chat_template(inference_messages, tokenize=False, add_generation_prompt=True)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"][0]
            tokens = [tokenizer.decode([tid]) for tid in input_ids]
            
            # Run forward pass intercepting attention tensors
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True
                )
            
            # Extract final layer attention weights
            # outputs.attentions shape: (layers, batch, heads, seq, seq)
            attention_matrix = outputs.attentions[-1][0] # Get final layer, remove batch
            mean_attention = attention_matrix.mean(dim=0).cpu().float().numpy() # Average over heads
            
            # Locate the prediction classification token (e.g. "AKI")
            target_word = "AKI"
            try:
                target_idx = [i for i, t in enumerate(tokens) if target_word in t][0]
            except IndexError:
                target_idx = len(tokens) - 2 # Fallback to second to last token
                
            # Slice the attention matrix specifically for the prediction token
            attention_slice = mean_attention[target_idx, :target_idx+1].reshape(1, -1)
            slice_tokens = tokens[:target_idx+1]

            # 4. Generate Heatmap
            plt.figure(figsize=(20, 5))
            sns.heatmap(
                attention_slice, 
                xticklabels=slice_tokens, 
                yticklabels=[f"Attention from '{tokens[target_idx].strip()}'"],
                cmap="Oranges", 
                cbar_kws={'label': 'Attention Weight'}, 
                square=True, 
                linewidths=0.5, 
                linecolor='gray'
            )
            
            plt.title(f"Patient {idx+1} Attention Mapping (Predicted: {tokens[target_idx].strip()})", fontsize=14, pad=15)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=10, fontweight='bold')
            
            out_path = os.path.join(out_dir, f"patient_{idx+1}_attention_heatmap.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            
    print(f"Pipeline complete! Saved all patient heatmaps to {out_dir}/")

if __name__ == "__main__":
    process_pipeline()
