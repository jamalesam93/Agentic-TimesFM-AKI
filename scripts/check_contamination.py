#!/usr/bin/env python3
"""
Check train/eval contamination: detect duplicate patient trajectories.

Compares training JSONL rows against eval holdout rows by normalizing the
user message (patient trajectory) text. Any verbatim overlap causes a hard fail.

Usage:
  python scripts/check_contamination.py output/dikd_training_data_10k.jsonl data/eval_holdout.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def norm(s: str) -> str:
    """Whitespace-normalize for comparison."""
    return " ".join(str(s).split()).lower()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def extract_user_text(record: dict) -> str:
    """Extract the user message content (patient trajectory) from a chat record."""
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def find_contamination(
    train_rows: list[dict], eval_rows: list[dict]
) -> list[tuple[int, int]]:
    """
    Returns pairs of (train_index, eval_index) where the user trajectory
    text is identical (after whitespace normalization).
    """
    # Build eval index: normalized text → eval row index
    eval_index: dict[str, int] = {}
    for i, row in enumerate(eval_rows):
        text = norm(extract_user_text(row))
        if text:
            eval_index[text] = i

    # Check each train row against the eval index
    hits: list[tuple[int, int]] = []
    for i, row in enumerate(train_rows):
        text = norm(extract_user_text(row))
        if text in eval_index:
            hits.append((i, eval_index[text]))

    return hits


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python scripts/check_contamination.py <train.jsonl> <eval.jsonl>", file=sys.stderr)
        return 1

    train_path = Path(sys.argv[1])
    eval_path = Path(sys.argv[2])

    if not train_path.exists():
        print(f"Train file not found: {train_path}", file=sys.stderr)
        return 1
    if not eval_path.exists():
        print(f"Eval file not found: {eval_path}", file=sys.stderr)
        return 1

    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)

    print(f"Checking {len(train_rows)} train rows vs {len(eval_rows)} eval rows")

    hits = find_contamination(train_rows, eval_rows)

    if hits:
        print(f"\nFAILED: {len(hits)} contaminated row(s) found!")
        for train_idx, eval_idx in hits[:10]:
            print(f"  Train row {train_idx} == Eval row {eval_idx}")
        return 1

    print("[SUCCESS] No contamination detected. Train and eval sets are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
