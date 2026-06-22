import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# =============================================================================
# DIKD FINE-TUNING PIPELINE — Gemma 4 12B IT (Vast.ai QLoRA)
#
# This script fine-tunes google/gemma-4-12b-it on 10,000 synthetic DIKD
# patient trajectories using QLoRA (4-bit Quantized Low-Rank Adaptation).
#
# ─── VAST.AI SETUP ───────────────────────────────────────────────────────────
#
#   1. Rent an instance with an NVIDIA A100 (40 GB) or RTX 4090 (24 GB).
#      Search template: PyTorch 2.x, CUDA 12.x, Ubuntu 22.04.
#
#   2. SSH into the instance and install dependencies:
#        pip install torch transformers datasets peft trl bitsandbytes accelerate
#
#   3. Upload the training data:
#        scp output/dikd_training_data_10k.jsonl root@<vast-ip>:~/data/
#
#   4. Upload this script:
#        scp src/hyperparameter_tuning.py root@<vast-ip>:~/
#
#   5. (Optional) If the model is gated, authenticate:
#        huggingface-cli login
#
#   6. Run training:
#        python hyperparameter_tuning.py
#
# ─── MEMORY BUDGET (4-bit QLoRA on A100 40 GB) ──────────────────────────────
#
#   Model (4-bit NF4):     ~7 GB
#   LoRA adapters:         ~0.3 GB
#   Optimizer (8-bit):     ~1.2 GB
#   Activations (bs=4):    ~6 GB
#   ────────────────────────────
#   TOTAL:                 ~15 GB  (fits comfortably on 24–40 GB GPUs)
#
# =============================================================================


def execute_training():
    # =========================================================================
    # 1. CONFIGURATION
    # =========================================================================
    model_id = "google/gemma-4-12b-it"
    dataset_path = "data/dikd_training_data_10k.jsonl"  # Path on the Vast.ai instance
    output_dir = "models/dikd-gemma4-12b-lora"

    print("=" * 65)
    print("     DIKD FINE-TUNING — Gemma 4 12B IT (Vast.ai QLoRA)")
    print("=" * 65)
    print(f"  Base Model     : {model_id}")
    print(f"  Dataset        : {dataset_path}")
    print(f"  Output         : {output_dir}")

    # Detect GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"  GPU            : {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        raise RuntimeError(
            "No GPU detected! Vast.ai instances should have a GPU. "
            "Check your rental or run `nvidia-smi` to diagnose."
        )
    print("-" * 65)

    # =========================================================================
    # 2. 4-BIT QUANTIZATION CONFIG (QLoRA)
    # =========================================================================
    # NF4 quantization compresses the 12B model from ~24 GB (bf16) to ~7 GB.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # =========================================================================
    # 3. LOAD MODEL & TOKENIZER
    # =========================================================================
    print("\n[1/5] Loading Gemma 4 12B with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",        # Safest default; switch to "flash_attention_2"
                                            # if flash-attn is installed for ~20% speedup
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # =========================================================================
    # 4. LoRA ADAPTER CONFIGURATION
    # =========================================================================
    # LoRA injects small trainable rank-decomposition matrices into the frozen
    # base model. Only ~0.4% of total parameters are actually trained.
    #
    #   r=16      → Rank-16 decomposition. With a 12B model and a structured
    #               clinical task, r=16 gives headroom for learning the full
    #               AKI decision boundary without overfitting.
    #   alpha=32  → Scaling factor. Standard rule of thumb: alpha = 2 * r.
    #   dropout=0.05 → Light dropout. The 12B base has strong representations;
    #                   we only need minimal regularization.
    #   target_modules → Gemma's multi-head attention projection layers.

    print("[2/5] Configuring LoRA adapters...")
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)

    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")

    # =========================================================================
    # 5. LOAD DATASET
    # =========================================================================
    print(f"\n[3/5] Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"  Loaded {len(dataset)} training sequences.")

    # =========================================================================
    # 6. TRAINING ARGUMENTS (Vast.ai GPU-Optimized)
    # =========================================================================
    # These settings target a single A100 40 GB or RTX 4090 24 GB.
    #
    #   batch_size=4 + grad_accum=4 → Effective batch size of 16.
    #     Good balance of throughput and gradient stability for a 10k dataset.
    #
    #   gradient_checkpointing=True → Even on A100, this prevents OOM on
    #     the 12B model. Trades ~25% speed for ~40% memory savings.
    #
    #   cosine scheduler → Smooth LR decay. Reaches near-zero by the end of
    #     training, which helps convergence on the final epoch.
    #
    #   3 epochs × 10k samples / 16 effective batch = 1,875 optimizer steps.
    #   Estimated time: ~45 min on A100, ~90 min on RTX 4090.

    print("[4/5] Configuring training hyperparameters...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        logging_steps=25,
        save_strategy="epoch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        fp16=False,
        bf16=True,
        report_to="none",
    )

    # =========================================================================
    # 7. EXECUTE TRAINING
    # =========================================================================
    # SFTTrainer parses the {"messages": [...]} chat format automatically and
    # applies the Gemma chat template. Loss is computed only on assistant turns.
    print("\n[5/5] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("=" * 65)
    print("  TRAINING STARTED")
    print("  Monitor GPU: watch -n 1 nvidia-smi")
    print("  Monitor logs: training loss logged every 25 steps")
    print("=" * 65)
    trainer.train()

    # =========================================================================
    # 8. SAVE TRAINED LoRA ADAPTERS
    # =========================================================================
    print(f"\n{'=' * 65}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 65}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  LoRA adapters saved to: {output_dir}/")
    print(f"  Adapter size: ~50–100 MB (only the trained delta weights)")
    print(f"\n  Next steps:")
    print(f"    1. Download the adapter folder from Vast.ai:")
    print(f"       scp -r root@<vast-ip>:~/{output_dir} ./models/")
    print(f"    2. Load locally in LM Studio or with HuggingFace:")
    print(f"       model = AutoModelForCausalLM.from_pretrained('{model_id}')")
    print(f"       model = PeftModel.from_pretrained(model, '{output_dir}')")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    execute_training()