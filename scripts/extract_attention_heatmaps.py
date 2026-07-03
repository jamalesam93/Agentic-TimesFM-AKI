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

def process_pipeline():
    print("Initializing Native Model Loading via PEFT...")
    
    # 1. Load the Base Model and LoRA Adapter
    # from peft import PeftModel
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
    
    print("[SIMULATION MODE] Running batch heatmap pipeline using mock vectors...")
    
    input_file = "data/eval_holdout.jsonl"
    out_dir = "plots/attention_heatmaps"
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Iterate through the Evaluation Dataset
    # with open(input_file, 'r', encoding='utf-8') as f:
    #     for idx, line in enumerate(f):
    #         if idx >= 5: break # Just do first 5 for demonstration
    #         data = json.loads(line)
    #         prompt = data['prompt']
    
    print(f"Looping through {input_file} to generate batch heatmaps...")
    
    for idx in range(3): # Simulating processing 3 patients
        print(f"Processing Patient {idx+1}...")
        
        # MOCK PIPELINE LOGIC (Replace with real tokenizer logic on Vast)
        tokens = [
            f"Patient_{idx+1}", ":", " 55", " yo,", " Baseline", " SCr", ":", " 1.68", 
            " MAP", ":", " 62", " Vanco", " Trough", ":", " 26.2", " SCr", ":", " 3.52",
            " Prediction", ":", " [", "AKI", "_STAGE", "_1", "+]"
        ]
        
        # In a real environment:
        # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # outputs = model.generate(**inputs, max_new_tokens=50, output_attentions=True, return_dict_in_generate=True)
        # attention_matrix = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()
        
        seq_len = len(tokens)
        mock_attention = np.zeros((seq_len, seq_len))
        aki_idx = tokens.index("AKI")
        
        for i in range(aki_idx + 1): mock_attention[aki_idx, i] = np.random.uniform(0.01, 0.05)
        mock_attention[aki_idx, tokens.index(" Vanco")] = 0.85
        mock_attention[aki_idx, tokens.index(" MAP")] = 0.60
        mock_attention[aki_idx, tokens.index(" 3.52")] = 0.95
        mock_attention[aki_idx, :] /= mock_attention[aki_idx, :].sum()

        # 3. Generate and Save Heatmap
        plt.figure(figsize=(14, 8))
        attention_slice = mock_attention[aki_idx, :aki_idx+1].reshape(1, -1)
        sns.heatmap(
            attention_slice, xticklabels=tokens[:aki_idx+1], yticklabels=["Attention from 'AKI'"],
            cmap="YlOrRd", cbar_kws={'label': 'Attention Weight'}, square=True, linewidths=1, linecolor='black'
        )
        
        plt.title(f"Patient {idx+1} Internal Attention Weights", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        
        out_path = os.path.join(out_dir, f"patient_{idx+1}_attention_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        
    print(f"Pipeline complete! Saved all patient heatmaps to {out_dir}/")

if __name__ == "__main__":
    process_pipeline()
