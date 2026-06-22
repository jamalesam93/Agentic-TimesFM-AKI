#!/usr/bin/env python3
"""
DIKD clinical quality tier gates — ship/no-ship decision.

Reads the metrics JSON produced by eval_dikd_batch.py and applies
clinical quality thresholds. A PASS means the model is safe to deploy
as an AKI early warning sentinel.

Gate rationale:
  - Sensitivity ≥ 95%: Missing a real AKI case is clinically dangerous.
    This is the hardest gate. A false negative means delayed intervention.
  - Specificity ≥ 85%: Too many false alarms cause alert fatigue.
    Clinicians will ignore the sentinel if it cries wolf constantly.
  - Parse rate ≥ 98%: Model must produce parseable structured output.
  - F1 ≥ 0.90: Overall balance between precision and recall.

Usage:
  python scripts/tier_gates.py reports/eval_predictions.metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Clinical Quality Gates ──────────────────────────────────────
GATES = {
    "sensitivity": {
        "min": 0.95,
        "label": "AKI Sensitivity (Recall)",
        "rationale": "Missing AKI = delayed intervention = patient harm",
    },
    "specificity": {
        "min": 0.85,
        "label": "Specificity (True Negative Rate)",
        "rationale": "Too many false alarms → alert fatigue → ignored sentinel",
    },
    "f1": {
        "min": 0.90,
        "label": "F1 Score",
        "rationale": "Overall precision-recall balance",
    },
    "parse_rate": {
        "min": 0.98,
        "label": "Structured Output Parse Rate",
        "rationale": "Model must produce machine-readable output",
    },
}


def evaluate_gates(metrics: dict) -> dict:
    results = {}
    all_passed = True

    for key, gate in GATES.items():
        value = metrics.get(key, 0)
        passed = value >= gate["min"]
        if not passed:
            all_passed = False
        results[key] = {
            "value": round(value, 4),
            "threshold": gate["min"],
            "passed": passed,
            "label": gate["label"],
        }

    return {
        "all_gates_passed": all_passed,
        "ship_decision": "GO" if all_passed else "NO-GO",
        "gates": results,
        "confusion": metrics.get("confusion", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DIKD clinical quality tier gates")
    parser.add_argument("metrics_file", type=Path, help="Metrics JSON from eval_dikd_batch.py")
    parser.add_argument("--json", type=Path, help="Output gate results to JSON")
    args = parser.parse_args()

    if not args.metrics_file.exists():
        print(f"Metrics file not found: {args.metrics_file}", file=sys.stderr)
        return 1

    metrics = json.loads(args.metrics_file.read_text(encoding="utf-8"))
    result = evaluate_gates(metrics)

    print(f"{'=' * 60}")
    print(f"  DIKD CLINICAL QUALITY GATES")
    print(f"{'=' * 60}")

    for key, gate in result["gates"].items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  {status}  {gate['label']}: {gate['value']:.1%} (min: {gate['threshold']:.0%})")

    print(f"{'-' * 60}")
    decision = result["ship_decision"]
    if decision == "GO":
        print(f"  DECISION: GO - Model meets all clinical safety gates.")
    else:
        print(f"  DECISION: NO-GO - One or more gates failed. Do NOT deploy.")

    cm = result["confusion"]
    if cm:
        print(f"\n  Confusion Matrix:")
        print(f"    TP={cm.get('tp', 0)}  FP={cm.get('fp', 0)}")
        print(f"    FN={cm.get('fn', 0)}  TN={cm.get('tn', 0)}")
    print(f"{'=' * 60}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nGate results saved to {args.json}")

    return 0 if result["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
