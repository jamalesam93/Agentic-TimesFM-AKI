import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def format_prompt(example):
    """
    Combines the system instruction, user prompt, and assistant response
    into a single string suitable for causal language modeling.
    """
    messages = example['messages']
    system_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
    user_msg = next((m['content'] for m in messages if m['role'] == 'user'), "")
    assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), "")
    
    # Simple formatting. In practice, you might use a specific chat template (e.g., chatml or gemma template).
    prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant: {assistant_msg}<eos>"
    return {"text": prompt}

def main():
    print("--- Phase 2: LLM Fine-Tuning Setup ---")
    
    # Configuration
    model_id = "google/gemma-12b"  # Base model as used in prior experiments
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "paper_sft_dataset.jsonl")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "clinical_sentinel_gemma")
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please run scripts/generate_llm_dataset.py first.")
        return

    print(f"1. Loading dataset from {data_path}...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Map messages to unified text format
    dataset = dataset.map(format_prompt, remove_columns=['messages'])
    
    print(f"2. Initializing Tokenizer and Model ({model_id})...")
    # Note: In a real run, you would likely use bitsandbytes for 4-bit quantization on a single GPU.
    # We load with dummy arguments here to illustrate the methodology.
    
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    
    print("3. Configuring LoRA (Low-Rank Adaptation)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # model = get_peft_model(model, peft_config)
    
    print("4. Preparing Training Arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        fp16=True,  # Use mixed precision
    )
    
    print("5. Initializing SFTTrainer...")
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset,
    #     peft_config=peft_config,
    #     dataset_text_field="text",
    #     max_seq_length=1024,
    #     tokenizer=tokenizer,
    #     args=training_args,
    # )
    
    print(f"\nTraining pipeline constructed successfully.")
    print("To execute training, uncomment the model initialization lines and run this script on a GPU instance.")

if __name__ == "__main__":
    main()
