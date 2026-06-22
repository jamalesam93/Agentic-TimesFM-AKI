#!/usr/bin/env python3
"""
Merge DIKD LoRA adapter into full Gemma 4 12B weights (bf16 HF shards).

Unsloth's save_pretrained_gguf / load_adapter+merge can fail on Gemma4's
ClippableLinear layers. This PeftModel path is the proven working recipe
from the Nassila pipeline.

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
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        print(f"Adapter not found: {args.adapter_dir}", file=sys.stderr)
        return 1

    try:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Requires unsloth + peft + torch on a GPU machine.\n"
            f"Original error: {e}"
        ) from e

    print(f"Loading base model: {args.base_model} (bf16, full precision)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )

    print(f"Loading adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.out_dir))

    print(f"\n[SUCCESS] Merged bf16 HF weights saved to {args.out_dir}")
    print("Next steps:")
    print(f"  1. python ~/llama.cpp/convert_hf_to_gguf.py {args.out_dir} "
          f"--outfile exports/dikd-f16.gguf --outtype f16")
    print(f"  2. ~/llama.cpp/build/bin/llama-quantize exports/dikd-f16.gguf "
          f"exports/dikd-q6_k.gguf Q6_K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
