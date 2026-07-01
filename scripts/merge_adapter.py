#!/usr/bin/env python3
"""
Merge DIKD LoRA adapter into full Gemma 4 12B weights (bf16 HF shards).

Tries Unsloth first (faster), falls back to vanilla PEFT + AutoModel
if Unsloth is not installed (e.g. training was done with --backend peft).

Usage (on Vast.ai GPU, after training):
  python scripts/merge_adapter.py \
    --adapter-dir outputs/dikd-gemma4-12b/lora_adapter \
    --out-dir exports/dikd-gemma4-12b-merged-bf16

Then convert with llama.cpp:
  python ~/llama.cpp/convert_hf_to_gguf.py exports/dikd-gemma4-12b-merged-bf16 \
    --outfile exports/dikd-gemma4-12b-f16.gguf --outtype f16
  ~/llama.cpp/build/bin/llama-quantize exports/dikd-gemma4-12b-f16.gguf \
    exports/dikd-gemma4-12b-q6_k.gguf Q6_K
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_MODEL = "google/gemma-4-12b-it"
MAX_SEQ_LENGTH = 512


def merge_with_unsloth(adapter_dir: Path, out_dir: Path, base_model: str, max_seq_length: int) -> None:
    """Merge using Unsloth (faster, handles ClippableLinear layers)."""
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from unsloth import FastLanguageModel  # type: ignore

    print(f"Loading base model: {base_model} (Unsloth, bf16)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )

    print(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"\n[SUCCESS] Merged bf16 HF weights saved to {out_dir}")


def merge_with_peft(adapter_dir: Path, out_dir: Path, base_model: str) -> None:
    """Fallback: merge using vanilla PEFT + AutoModel (no Unsloth required)."""
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    print(f"Loading base model: {base_model} (AutoModel, bf16)...")
    print("  NOTE: This requires ~24GB VRAM for a 12B model in bf16.")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"\n[SUCCESS] Merged bf16 HF weights saved to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge DIKD LoRA adapter into bf16 HF weights")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help=f"HF base model id (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "unsloth", "peft"),
        default="auto",
        help="Merge backend: auto (try unsloth, fall back to peft), unsloth, or peft",
    )
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        print(f"Adapter not found: {args.adapter_dir}", file=sys.stderr)
        return 1

    # Check that required packages are available
    try:
        import torch  # type: ignore  # noqa: F401
        from peft import PeftModel  # type: ignore  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Requires peft + torch on a GPU machine.\n"
            f"Original error: {e}"
        ) from e

    # Determine backend
    use_unsloth = False
    if args.backend in ("auto", "unsloth"):
        try:
            from unsloth import FastLanguageModel  # type: ignore  # noqa: F401
            use_unsloth = True
        except ImportError:
            if args.backend == "unsloth":
                raise SystemExit("Unsloth not installed. Use --backend peft or --backend auto.")
            print("Unsloth not available, falling back to PEFT merge path.")

    if use_unsloth:
        merge_with_unsloth(args.adapter_dir, args.out_dir, args.base_model, args.max_seq_length)
    else:
        merge_with_peft(args.adapter_dir, args.out_dir, args.base_model)

    print("Next steps:")
    print(f"  1. python ~/llama.cpp/convert_hf_to_gguf.py {args.out_dir} "
          f"--outfile exports/dikd-f16.gguf --outtype f16")
    print(f"  2. ~/llama.cpp/build/bin/llama-quantize exports/dikd-f16.gguf "
          f"exports/dikd-q6_k.gguf Q6_K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
