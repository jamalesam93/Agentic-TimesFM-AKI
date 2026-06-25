#!/usr/bin/env python3
"""
Audit label distribution in DIKD training data.

Reports AKI vs NORMAL class balance and flags imbalanced datasets.
Writes a JSON summary for pipeline consumption.

Usage:
  python scripts/audit_labels.py output/dikd_training_data_10k.jsonl --json reports/audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LABEL_RE = re.compile(r"\[(AKI_STAGE_1\+|NORMAL)\]")

# Acceptable class balance range (AKI proportion)
MIN_AKI_RATIO = 0.10  # At least 10% AKI cases
MAX_AKI_RATIO = 0.55  # At most 55% AKI cases (allows balanced SFT datasets)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def extract_label(record: dict) -> str:
    messages = record.get("messages", [])
    if len(messages) >= 3:
        m = LABEL_RE.search(messages[2].get("content", ""))
        if m:
            return m.group(1)
    return "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit DIKD label distribution")
    parser.add_argument("path", type=Path, help="Training JSONL file")
    parser.add_argument("--json", type=Path, help="Output JSON summary")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1

    rows = load_jsonl(args.path)
    labels = [extract_label(r) for r in rows]

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    total = len(rows)
    aki = counts.get("AKI_STAGE_1+", 0)
    normal = counts.get("NORMAL", 0)
    unknown = counts.get("UNKNOWN", 0)
    aki_ratio = aki / total if total else 0

    print(f"{'=' * 50}")
    print(f"  DIKD Label Audit: {args.path.name}")
    print(f"{'=' * 50}")
    print(f"  Total        : {total}")
    print(f"  AKI_STAGE_1+ : {aki:>6} ({100 * aki / total:.1f}%)")
    print(f"  NORMAL       : {normal:>6} ({100 * normal / total:.1f}%)")
    if unknown:
        print(f"  UNKNOWN      : {unknown:>6} ({100 * unknown / total:.1f}%)")
    print(f"{'=' * 50}")

    # Balance check
    balanced = MIN_AKI_RATIO <= aki_ratio <= MAX_AKI_RATIO
    if not balanced:
        print(
            f"\n⚠ WARNING: AKI ratio {aki_ratio:.2%} outside acceptable range "
            f"[{MIN_AKI_RATIO:.0%}–{MAX_AKI_RATIO:.0%}]"
        )
    else:
        print(f"\n[SUCCESS] Class balance OK (AKI ratio: {aki_ratio:.2%})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "file": str(args.path),
            "total": total,
            "counts": counts,
            "aki_ratio": round(aki_ratio, 4),
            "balanced": balanced,
        }
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {args.json}")

    return 0 if balanced and unknown == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
