#!/usr/bin/env python3
"""
Compare MIMIC-IV real-world vs synthetic evaluation results side-by-side.

Reads the .metrics.json files from both evaluations and produces a
structured comparison report (JSON + pretty console output).

Usage:
  python scripts/compare_eval_results.py \
    --mimic-metrics reports/mimic_eval_predictions.metrics.json \
    --synth-metrics reports/synth_eval_predictions.metrics.json \
    --out reports/mimic_vs_synth_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MIMIC vs Synthetic eval metrics")
    parser.add_argument("--mimic-metrics", type=Path, required=True)
    parser.add_argument("--synth-metrics", type=Path, required=False, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mimic = load_metrics(args.mimic_metrics)
    synth = load_metrics(args.synth_metrics) if args.synth_metrics else None

    if not mimic:
        print(f"Error: MIMIC metrics not found at {args.mimic_metrics}", file=sys.stderr)
        return 1

    report = {
        "mimic_iv": mimic,
        "synthetic": synth,
        "comparison": {},
    }

    # Build comparison
    metrics_to_compare = ["accuracy", "sensitivity", "specificity", "precision", "f1", "parse_rate"]
    for metric in metrics_to_compare:
        mimic_val = mimic.get(metric, 0)
        entry = {"mimic_iv": mimic_val}
        if synth:
            synth_val = synth.get(metric, 0)
            entry["synthetic"] = synth_val
            entry["delta"] = round(mimic_val - synth_val, 4)
        report["comparison"][metric] = entry

    # Pretty print
    print()
    print("=" * 64)
    print("  MIMIC-IV vs SYNTHETIC — EVALUATION COMPARISON")
    print("=" * 64)

    header = f"  {'Metric':<20} {'MIMIC-IV':>12}"
    if synth:
        header += f" {'Synthetic':>12} {'Delta':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for metric in metrics_to_compare:
        mimic_val = mimic.get(metric, 0)
        line = f"  {metric:<20} {mimic_val:>11.1%}"
        if synth:
            synth_val = synth.get(metric, 0)
            delta = mimic_val - synth_val
            sign = "+" if delta >= 0 else ""
            line += f" {synth_val:>11.1%} {sign}{delta:>8.1%}"
        print(line)

    print("  " + "─" * (len(header) - 2))

    # Confusion matrix
    cm = mimic.get("confusion", {})
    print(f"\n  MIMIC-IV Confusion Matrix:")
    print(f"    TP={cm.get('tp', 0)}  FP={cm.get('fp', 0)}")
    print(f"    FN={cm.get('fn', 0)}  TN={cm.get('tn', 0)}")

    if synth:
        cm2 = synth.get("confusion", {})
        print(f"\n  Synthetic Confusion Matrix:")
        print(f"    TP={cm2.get('tp', 0)}  FP={cm2.get('fp', 0)}")
        print(f"    FN={cm2.get('fn', 0)}  TN={cm2.get('tn', 0)}")

    print("=" * 64)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nComparison saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
