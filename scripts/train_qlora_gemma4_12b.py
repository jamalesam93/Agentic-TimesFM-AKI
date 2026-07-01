#!/usr/bin/env python3
"""
QLoRA fine-tuning for DIKD AKI sentinel on Gemma 4 12B IT.

Unsloth-first with PEFT fallback. Adapted from the Nassila pipeline.

IMPORTANT:
  - save_strategy="no": Unsloth + Gemma 4 pickles crash on Vast.ai
    when trying to serialize mid-training checkpoints via Trainer.
    We work around this with a custom LoRACheckpointCallback that
    saves only the lightweight adapter weights every N steps.
  - 7 LoRA target modules: attention projections (q/k/v/o) + MLP
    (gate/up/down) for richer adaptation (~1% trainable params).
  - apply_chat_template(): ensures the model sees the exact Gemma chat
    format during training, not raw JSON strings.

Usage (on Vast.ai after git clone):
  pip install unsloth torch transformers datasets trl bitsandbytes accelerate
  python scripts/train_qlora_gemma4_12b.py
  python scripts/train_qlora_gemma4_12b.py --backend peft  # fallback if unsloth fails
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time
from pathlib import Path

# Disable JIT/Dynamo recompilation checks and Unsloth graph compile overhead
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import torch
    torch._dynamo.config.cache_size_limit = 256
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# --- Configuration ---
BASE_MODEL = "google/gemma-4-12b-it"
MAX_SEQ_LENGTH = 512               # DIKD trajectories are ~350 tokens
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 2
GRAD_ACCUM = 8                     # Effective batch = 16
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
CHECKPOINT_STEPS = 100             # Save LoRA adapter every N steps

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP FFN
]


# ── Custom callback: saves LoRA adapter weights periodically ────
# This bypasses Trainer's checkpoint serialization (which crashes
# Unsloth + Gemma 4 on Vast.ai) by saving only the small adapter
# weights via model.save_pretrained().
class LoRACheckpointCallback:
    """TrainerCallback that saves LoRA adapter weights every N steps.

    Avoids the Unsloth/Gemma4 pickle crash by never going through
    Trainer's full checkpoint serialization path.
    """

    def __init__(self, save_steps: int, output_dir: Path):
        self.save_steps = save_steps
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if (
            self.save_steps > 0
            and state.global_step > 0
            and state.global_step % self.save_steps == 0
        ):
            ckpt_dir = self.output_dir / f"lora_checkpoint_step_{state.global_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            print(f"  [Checkpoint] LoRA adapter saved at step {state.global_step} → {ckpt_dir.name}")


class TrainingLogger:
    """TrainerCallback that logs training metrics to a JSON Lines file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear any existing log
        self.log_path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {
                "step": state.global_step,
                "epoch": round(state.epoch, 4) if state.epoch else 0,
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()},
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")


def _make_callbacks(output_dir: Path, save_steps: int):
    """Create the list of trainer callbacks."""
    from transformers import TrainerCallback  # type: ignore

    # Inherit from TrainerCallback so the Trainer recognizes them
    class _LoRACheckpoint(TrainerCallback, LoRACheckpointCallback):
        def __init__(self, save_steps, output_dir):
            LoRACheckpointCallback.__init__(self, save_steps, output_dir)

    class _TrainingLogger(TrainerCallback, TrainingLogger):
        def __init__(self, log_path):
            TrainingLogger.__init__(self, log_path)

    callbacks = []
    if save_steps > 0:
        callbacks.append(_LoRACheckpoint(save_steps, output_dir))
    callbacks.append(_TrainingLogger(output_dir / "training_log.jsonl"))
    return callbacks


def train_with_unsloth(
    train_file: Path,
    output_dir: Path,
    *,
    num_epochs: int,
    learning_rate: float,
    save_steps: int,
) -> None:
    """Train using Unsloth (2x faster, 60% less VRAM)."""
    try:
        from unsloth import FastLanguageModel  # type: ignore
        from trl import SFTTrainer  # type: ignore
        from transformers import TrainingArguments  # type: ignore
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Missing training deps. Install: pip install unsloth trl transformers datasets\n"
            f"Original error: {e}"
        ) from e

    print(f"\n[1/4] Loading {BASE_MODEL} with Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Ensure padding is configured for variable-length batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[2/4] Configuring LoRA adapters (7 target modules)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
    )

    print(f"[3/4] Loading dataset from {train_file}...")
    dataset = load_dataset("json", data_files=str(train_file), split="train")

    # Apply Gemma chat template — this is critical for correct tokenization
    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)
    print(f"  Loaded {len(dataset)} training sequences.")

    # save_strategy="no": Unsloth/Gemma4 crashes on checkpoint pickle on Vast.
    # We use LoRACheckpointCallback instead to save adapter weights periodically.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        logging_steps=25,
        fp16=False,
        bf16=True,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=training_args,
        callbacks=_make_callbacks(output_dir, save_steps),
    )

    print(f"\n[4/4] Training started (epochs={num_epochs}, lr={learning_rate}, "
          f"batch={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE * GRAD_ACCUM})...")
    print(f"  LoRA checkpoints every {save_steps} steps → {output_dir}")
    print("  Monitor GPU: watch -n 1 nvidia-smi")
    trainer.train()

    # Save final adapter
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\n[SUCCESS] Saved LoRA adapter to {adapter_dir}")


def train_with_peft(
    train_file: Path,
    output_dir: Path,
    *,
    num_epochs: int,
    learning_rate: float,
    save_steps: int,
) -> None:
    """Fallback: train using vanilla PEFT + BitsAndBytes (if Unsloth unavailable)."""
    try:
        import torch  # type: ignore
        from datasets import load_dataset  # type: ignore
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer  # type: ignore
    except ImportError as e:
        raise SystemExit(
            f"Missing torch/peft/transformers/trl/datasets.\nOriginal error: {e}"
        ) from e

    print(f"\n[1/4] Loading {BASE_MODEL} with PEFT (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    print("[2/4] Configuring LoRA adapters (7 target modules)...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print(f"[3/4] Loading dataset from {train_file}...")
    dataset = load_dataset("json", data_files=str(train_file), split="train")

    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)
    print(f"  Loaded {len(dataset)} training sequences.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        logging_steps=25,
        bf16=True,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=training_args,
        callbacks=_make_callbacks(output_dir, save_steps),
    )

    print(f"\n[4/4] Training started (epochs={num_epochs}, lr={learning_rate}, "
          f"batch={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE * GRAD_ACCUM})...")
    print(f"  LoRA checkpoints every {save_steps} steps → {output_dir}")
    trainer.train()

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\n[SUCCESS] Saved LoRA adapter to {adapter_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="DIKD QLoRA fine-tuning on Gemma 4 12B")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=PROJECT_DIR / "output" / "llm_fine_tuning_dataset.jsonl",
        help="Training JSONL file (default: output/llm_fine_tuning_dataset.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "dikd-gemma4-12b",
        help="Output directory for adapter and logs",
    )
    parser.add_argument(
        "--backend",
        choices=("unsloth", "peft"),
        default="unsloth",
        help="Training backend: unsloth (default, faster) or peft (fallback)",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=CHECKPOINT_STEPS,
        help=f"Save LoRA adapter checkpoint every N steps (default: {CHECKPOINT_STEPS}, 0=disabled)",
    )
    args = parser.parse_args()

    if not args.train_file.exists():
        print(f"Training file not found: {args.train_file}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DIKD FINE-TUNING — Gemma 4 12B IT (QLoRA)")
    print("=" * 60)
    print(f"  Base Model  : {BASE_MODEL}")
    print(f"  Backend     : {args.backend}")
    print(f"  Train File  : {args.train_file}")
    print(f"  Output Dir  : {args.output_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  LoRA        : r={LORA_R}, α={LORA_ALPHA}, modules={len(TARGET_MODULES)}")
    print(f"  Seq Length  : {MAX_SEQ_LENGTH}")
    print(f"  Checkpoints : every {args.save_steps} steps" if args.save_steps > 0 else "  Checkpoints : disabled")
    print("-" * 60)

    if args.backend == "unsloth":
        train_with_unsloth(
            args.train_file,
            args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            save_steps=args.save_steps,
        )
    else:
        train_with_peft(
            args.train_file,
            args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            save_steps=args.save_steps,
        )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Adapter saved to: {args.output_dir / 'lora_adapter'}")
    print(f"  Next: python scripts/merge_adapter.py --adapter-dir {args.output_dir / 'lora_adapter'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
