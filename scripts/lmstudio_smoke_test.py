#!/usr/bin/env python3
"""
LM Studio smoke test for the DIKD AKI sentinel.

Loads the first patient trajectory from the evaluation holdout, sends it
to a running LM Studio (or llama-server) instance, and displays the response.
Verifies that the response format and tags are correct.

Usage:
  python scripts/lmstudio_smoke_test.py \
    --base-url http://127.0.0.1:1234 \
    --model dikd-gemma4-12b \
    --data data/eval_holdout.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    raise SystemExit(1)

LABEL_RE = re.compile(r"\[(AKI_STAGE_1\+|NORMAL)\]")

def main() -> int:
    parser = argparse.ArgumentParser(description="DIKD LM Studio smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--model", default="dikd-gemma4-12b")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/eval_holdout.jsonl"),
        help="Eval holdout file to draw a sample from",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Data file not found: {args.data}. Make sure to run build_holdout.py first.", file=sys.stderr)
        return 1

    # Load first record
    with args.data.open(encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            print("Empty data file", file=sys.stderr)
            return 1
        record = json.loads(first_line)

    messages = record["messages"]
    user_prompt = messages[0]["content"]
    gt_content = messages[1]["content"]

    print("=" * 60)
    print("  DIKD SENTINEL SMOKE TEST")
    print("=" * 60)
    print(f"  Target Endpoint: {args.base_url}")
    print(f"  Target Model   : {args.model}")
    print("-" * 60)
    print("  User Prompt Summary:")
    print(f"    {user_prompt[:120]}...")
    print("\n  Patient Trajectory:")
    print(user_prompt)
    print("-" * 60)
    print("  Ground Truth Assistant Response:")
    print(gt_content)
    print("=" * 60)

    # Call API
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }

    print("Sending request...")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}", file=sys.stderr)
        return 1

    print("\nModel Response:")
    print(response_text)
    print("-" * 60)

    # Extract label
    label = LABEL_RE.search(response_text)
    if label:
        pred_label = label.group(1)
        gt_label = LABEL_RE.search(gt_content).group(1) if LABEL_RE.search(gt_content) else "UNKNOWN"
        print(f"Parsed Prediction: [{pred_label}]")
        print(f"Ground Truth     : [{gt_label}]")
        if pred_label == gt_label:
            print("\n[SUCCESS] Smoke test passed! Model predicted correctly and output format is valid.")
            return 0
        else:
            print("\n[WARNING] Smoke test completed. Model format is valid, but prediction differs from ground truth.")
            return 0
    else:
        print("\n[FAILED] Model response does not contain the required [AKI_STAGE_1+] or [NORMAL] tag.", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
