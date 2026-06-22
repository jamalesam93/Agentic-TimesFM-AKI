import json
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# SYNTHETIC DIKD DATA LOADER & TOKEN ORACLE
# This script ingests the JSONL tapestries, measures their token weight, 
# and prepares the sequences for the model's consumption.
# -----------------------------------------------------------------------------

def load_jsonl_dataset(filepath: str) -> List[Dict]:
    """Reads the somber chronicles of the synthetic cohort."""
    dataset = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        print(f"[+] Successfully resurrected {len(dataset)} patient trajectories from {filepath}.")
        return dataset
    except FileNotFoundError:
        print(f"[-] The archive {filepath} remains sealed or missing.")
        return []

def analyze_token_burden(dataset: List[Dict], tokenizer_id: str = "meta-llama/Meta-Llama-3-8B"):
    """
    Measures the exact token length of each patient's timeline to ensure 
    they fit within the model's context window.
    """
    print(f"\n[~] Summoning tokenizer: {tokenizer_id}...")
    try:
        # Using a fast tokenizer to measure the weight of the words
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as e:
        print(f"[-] Failed to load gated tokenizer '{tokenizer_id}'. Error: {e}")
        print(f"[TIP]: When running on Vast.ai, log in using 'huggingface-cli login' or export your HuggingFace token as an environment variable: 'export HF_TOKEN=your_token_here'")
        fallback_id = "openai-community/gpt2"
        print(f"[~] Falling back to standard non-gated tokenizer: {fallback_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(fallback_id)
        except Exception as fallback_err:
            print(f"[-] Failed to load fallback tokenizer. Error: {fallback_err}")
            return

    token_lengths = []
    risk_distribution = {"NORMAL": 0, "AKI_STAGE_1+": 0}

    for record in dataset:
        # Reconstruct the full sequence from the structured messages
        full_text = ""
        for msg in record.get("messages", []):
            full_text += f"{msg['role']}: {msg['content']}\n"
        
        # Tokenize and count
        tokens = tokenizer.encode(full_text, add_special_tokens=True)
        token_lengths.append(len(tokens))

        # Extract the final clinical state for class balancing
        if "AKI_STAGE_1+" in full_text:
            risk_distribution["AKI_STAGE_1+"] += 1
        elif "NORMAL" in full_text:
            risk_distribution["NORMAL"] += 1

    # Calculate the statistical weight of the text
    print("\n--- CORPUS ANALYSIS ---")
    print(f"Total Sequences Analyzed:  {len(token_lengths)}")
    print(f"Average Token Length:      {np.mean(token_lengths):.1f} tokens")
    print(f"Maximum Token Length:      {np.max(token_lengths)} tokens")
    print(f"Minimum Token Length:      {np.min(token_lengths)} tokens")
    print(f"95th Percentile Length:    {np.percentile(token_lengths, 95):.1f} tokens")
    
    print("\n--- CLINICAL CLASS BALANCE ---")
    total_labeled = sum(risk_distribution.values())
    for risk_state, count in risk_distribution.items():
        percentage = (count / total_labeled) * 100 if total_labeled > 0 else 0
        print(f"{risk_state}: {count} sequences ({percentage:.1f}%)")

    # 🛑 REALISM CHECK: Warn if the classes are wildly imbalanced
    if risk_distribution["AKI_STAGE_1+"] / total_labeled < 0.10:
        print("\n[WARNING]: Severe class imbalance detected. The model may struggle to recognize the rare AKI events. Consider adjusting the synergistic toxicity rates in the extraction phase.")

    return token_lengths, risk_distribution

if __name__ == "__main__":
    # Adjust this path to wherever Antigravity placed the 10k output
    TARGET_FILE = "output/dikd_training_data_10k.jsonl"
    
    cohort_data = load_jsonl_dataset(TARGET_FILE)
    if cohort_data:
        analyze_token_burden(cohort_data)