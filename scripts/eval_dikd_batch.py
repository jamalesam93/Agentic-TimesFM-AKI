#!/usr/bin/env python3
"""
Batch evaluation for the DIKD AKI sentinel model.

Sends holdout patient trajectories to a running llama-server (or any
OpenAI-compatible API) and scores the responses against ground truth.

Calculates:
  - Sensitivity (recall for AKI class — the critical metric)
  - Specificity (true negative rate for NORMAL class)
  - Precision, F1, accuracy
  - JSON parse success rate

Usage:
  python scripts/eval_dikd_batch.py \
    --base-url http://127.0.0.1:1234 \
    --model dikd-gemma4-12b \
    --data data/eval_holdout.jsonl \
    --out reports/eval_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    raise SystemExit(1)

LABEL_RE = re.compile(r"\[(AKI_STAGE_1\+|NORMAL|AKI_IMMINENT)\]")


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    api_key: str = "lm-studio",
    temperature: float = 0.1,
    timeout: int = 180,
) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    return content


def extract_label(text: str) -> str | None:
    m = LABEL_RE.search(text)
    if m:
        label = m.group(1)
        if label == "AKI_IMMINENT":
            return "AKI_STAGE_1+"
        return label
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="DIKD batch evaluation")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--model", required=True, help="Model name for API")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        required=True,
        help="One or more eval JSONL files",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output predictions JSONL")
    parser.add_argument("--retry", type=int, default=1, help="Retries on API failure")
    args = parser.parse_args()

    # Load eval data
    eval_rows: list[dict] = []
    for data_file in args.data:
        if not data_file.exists():
            print(f"Eval file not found: {data_file}", file=sys.stderr)
            return 1
        with data_file.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    eval_rows.append(json.loads(line.strip()))

    print(f"{'=' * 60}")
    print(f"  DIKD Batch Evaluation")
    print(f"{'=' * 60}")
    print(f"  Model    : {args.model}")
    print(f"  Endpoint : {args.base_url}")
    print(f"  Eval rows: {len(eval_rows)}")
    print(f"  Output   : {args.out}")
    print(f"{'=' * 60}")

    # Run inference
    predictions: list[dict] = []
    tp = fp = tn = fn = 0
    parse_ok = parse_fail = 0

    for i, row in enumerate(eval_rows):
        messages = row["messages"]
        # Ground truth from the assistant message
        gt_label = extract_label(messages[2]["content"])

        # Send only system + user messages (no assistant — model must predict)
        inference_messages = [
            {"role": messages[0]["role"], "content": messages[0]["content"]},
            {"role": messages[1]["role"], "content": messages[1]["content"]},
        ]

        raw_response = ""
        for attempt in range(args.retry + 1):
            try:
                raw_response = chat_completion(
                    args.base_url, args.model, inference_messages, args.api_key
                )
                break
            except Exception as e:
                if attempt == args.retry:
                    print(f"  [{i + 1}/{len(eval_rows)}] FAILED after {args.retry + 1} attempts: {e}")
                    raw_response = ""
                else:
                    time.sleep(2)

        pred_label = extract_label(raw_response)

        if pred_label:
            parse_ok += 1
        else:
            parse_fail += 1
            pred_label = "PARSE_FAIL"

        # Confusion matrix (AKI = positive class)
        # PARSE_FAIL is treated conservatively:
        #   - gt=AKI + PARSE_FAIL → FN (missed real AKI — clinically dangerous)
        #   - gt=NORMAL + PARSE_FAIL → FP (system failure → counts against specificity)
        if gt_label == "AKI_STAGE_1+" and pred_label == "AKI_STAGE_1+":
            tp += 1
        elif gt_label == "NORMAL" and pred_label == "NORMAL":
            tn += 1
        elif gt_label == "NORMAL" and pred_label != "NORMAL":
            fp += 1
        elif gt_label == "AKI_STAGE_1+" and pred_label != "AKI_STAGE_1+":
            fn += 1

        predictions.append({
            "index": i,
            "ground_truth": gt_label,
            "predicted": pred_label,
            "raw_response": raw_response[:500],
            "correct": gt_label == pred_label,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(eval_rows)}] processed...")

    # Save predictions
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    # Calculate metrics
    total = len(eval_rows)
    correct = sum(1 for p in predictions if p["correct"])
    accuracy = correct / total if total else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0  # Recall for AKI
    specificity = tn / (tn + fp) if (tn + fp) else 0  # True negative rate
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0
    parse_rate = parse_ok / total if total else 0

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total        : {total}")
    print(f"  Correct      : {correct} ({accuracy:.1%})")
    print(f"  Parse rate   : {parse_ok}/{total} ({parse_rate:.1%})")
    print(f"  ─────────────────────────────────")
    print(f"  Sensitivity  : {sensitivity:.1%}  (TP={tp}, FN={fn})")
    print(f"  Specificity  : {specificity:.1%}  (TN={tn}, FP={fp})")
    print(f"  Precision    : {precision:.1%}")
    print(f"  F1 Score     : {f1:.3f}")
    print(f"{'=' * 60}")

    # Write metrics summary
    metrics = {
        "total": total,
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "parse_rate": round(parse_rate, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Predictions saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
