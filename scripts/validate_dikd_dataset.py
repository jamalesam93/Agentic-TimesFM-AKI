#!/usr/bin/env python3
"""
Validate DIKD training JSONL dataset.

Checks:
  - Every line is valid JSON with a "messages" array
  - Messages follow system → user → assistant role ordering
  - Assistant response contains [AKI_STAGE_1+] or [NORMAL]
  - (Optional) No sequence exceeds --strict-length tokens
  - Reports class balance (NORMAL vs AKI)

Usage:
  python scripts/validate_dikd_dataset.py output/dikd_training_data_10k.jsonl
  python scripts/validate_dikd_dataset.py output/dikd_training_data_10k.jsonl --strict-length 512
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VALID_LABELS = {"AKI_STAGE_1+", "NORMAL"}
LABEL_RE = re.compile(r"\[(AKI_STAGE_1\+|NORMAL)\]")

EXPECTED_ROLES = ("user", "assistant")


def load_jsonl(path: Path) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append((line_no, json.loads(stripped)))
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no}: invalid JSON: {e}") from e
    return rows


def extract_label(assistant_content: str) -> str | None:
    """Extract the [LABEL] tag from the assistant response."""
    m = LABEL_RE.search(assistant_content)
    return m.group(1) if m else None


def validate_row(record: dict, line_no: int) -> list[str]:
    errors: list[str] = []

    messages = record.get("messages")
    if not isinstance(messages, list):
        errors.append(f"Line {line_no}: 'messages' must be an array")
        return errors

    if len(messages) != 2:
        errors.append(f"Line {line_no}: expected 2 messages (user/assistant), got {len(messages)}")
        return errors

    for i, expected_role in enumerate(EXPECTED_ROLES):
        msg = messages[i]
        if not isinstance(msg, dict):
            errors.append(f"Line {line_no}: messages[{i}] must be an object")
            continue
        role = msg.get("role")
        if role != expected_role:
            errors.append(f"Line {line_no}: messages[{i}] role='{role}', expected '{expected_role}'")
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"Line {line_no}: messages[{i}] has empty content")

    # Validate assistant label
    if len(messages) == 2 and isinstance(messages[1], dict):
        assistant = messages[1].get("content", "")
        label = extract_label(assistant)
        if label is None:
            errors.append(f"Line {line_no}: assistant response missing [AKI_STAGE_1+] or [NORMAL] tag")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DIKD training JSONL")
    parser.add_argument("path", type=Path, help="Path to .jsonl file")
    parser.add_argument(
        "--strict-length",
        type=int,
        metavar="MAX_TOKENS",
        help="Fail if any row exceeds this token count (requires transformers)",
    )
    parser.add_argument("--json", type=Path, help="Write validation summary to JSON file")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1

    rows = load_jsonl(args.path)
    all_errors: list[str] = []
    label_counts: dict[str, int] = {"AKI_STAGE_1+": 0, "NORMAL": 0, "UNKNOWN": 0}

    for line_no, record in rows:
        if not isinstance(record, dict):
            all_errors.append(f"Line {line_no}: record must be JSON object")
            continue
        all_errors.extend(validate_row(record, line_no))

        # Count labels
        messages = record.get("messages", [])
        if len(messages) == 2:
            label = extract_label(messages[1].get("content", ""))
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
            else:
                label_counts["UNKNOWN"] += 1

    # Sequence length check
    length_errors: list[str] = []
    if args.strict_length:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-4-12b-it", trust_remote_code=True
            )
            for line_no, record in rows:
                text = tokenizer.apply_chat_template(
                    record["messages"], tokenize=False, add_generation_prompt=False
                )
                n_tok = len(tokenizer.encode(text, add_special_tokens=False))
                if n_tok > args.strict_length:
                    length_errors.append(f"Line {line_no}: {n_tok} tokens > {args.strict_length}")
        except ImportError:
            print("strict-length requires transformers; skipping", file=sys.stderr)

    # Report
    total = len(rows)
    aki_count = label_counts.get("AKI_STAGE_1+", 0)
    normal_count = label_counts.get("NORMAL", 0)
    aki_pct = 100 * aki_count / total if total else 0
    normal_pct = 100 * normal_count / total if total else 0

    print(f"{'=' * 55}")
    print(f"  DIKD Dataset Validation: {args.path.name}")
    print(f"{'=' * 55}")
    print(f"  Total rows     : {total}")
    print(f"  AKI_STAGE_1+   : {aki_count} ({aki_pct:.1f}%)")
    print(f"  NORMAL         : {normal_count} ({normal_pct:.1f}%)")
    print(f"  Schema errors  : {len(all_errors)}")
    print(f"  Length overflows: {len(length_errors)}")
    print(f"{'=' * 55}")

    if all_errors:
        print(f"\nFAILED: {len(all_errors)} schema error(s):")
        for err in all_errors[:20]:
            print(f"  - {err}")

    if length_errors:
        print(f"\nFAILED: {len(length_errors)} length overflow(s):")
        for err in length_errors[:20]:
            print(f"  - {err}")

    # Write summary JSON
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "file": str(args.path),
            "total_rows": total,
            "label_counts": label_counts,
            "schema_errors": len(all_errors),
            "length_overflows": len(length_errors),
            "passed": len(all_errors) == 0 and len(length_errors) == 0,
        }
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSummary written to {args.json}")

    if all_errors or length_errors:
        return 1

    print("\n[SUCCESS] All validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
