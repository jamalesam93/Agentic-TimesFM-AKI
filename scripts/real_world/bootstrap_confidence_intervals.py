#!/usr/bin/env python3
"""
bootstrap_confidence_intervals.py — 95% Bootstrap CIs for Article Metrics

Reads the real-world evaluation predictions and computes bootstrap
confidence intervals for accuracy, sensitivity, specificity, precision,
and F1 score. Outputs a JSON file and a formatted table for the article.

Usage:
    python scripts/real_world/bootstrap_confidence_intervals.py
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

N_BOOTSTRAP = 10000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95


def load_predictions(filepath):
    """Load evaluation predictions and extract (y_true, y_pred) pairs."""
    y_true = []
    y_pred = []
    
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # The eval predictions contain 'label' (ground truth) and 'predicted' fields
            label = record.get('label') or record.get('ground_truth') or record.get('true_label')
            pred = record.get('predicted') or record.get('prediction') or record.get('pred_label')
            
            if label is None or pred is None:
                continue
            
            # Normalize labels to binary
            is_aki_true = 1 if 'AKI' in str(label).upper() and 'NORMAL' not in str(label).upper() else 0
            is_aki_pred = 1 if 'AKI' in str(pred).upper() and 'NORMAL' not in str(pred).upper() else 0
            
            y_true.append(is_aki_true)
            y_pred.append(is_aki_pred)
    
    return np.array(y_true), np.array(y_pred)


def compute_metrics(y_true, y_pred):
    """Compute binary classification metrics from arrays."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
    }


def bootstrap_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, confidence=CONFIDENCE_LEVEL, seed=RANDOM_SEED):
    """
    Compute bootstrap confidence intervals for all classification metrics.
    
    Uses the percentile method: draw n_bootstrap resamples with replacement,
    compute metrics on each, then take the (α/2, 1-α/2) percentiles.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = 1 - confidence
    
    # Storage for bootstrap samples
    boot_metrics = {k: [] for k in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1']}
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.randint(0, n, size=n)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        metrics = compute_metrics(y_true_boot, y_pred_boot)
        for k, v in metrics.items():
            boot_metrics[k].append(v)
    
    # Compute point estimates and CIs
    point_estimates = compute_metrics(y_true, y_pred)
    results = {}
    
    for metric_name in boot_metrics:
        values = np.array(boot_metrics[metric_name])
        lo = np.percentile(values, 100 * alpha / 2)
        hi = np.percentile(values, 100 * (1 - alpha / 2))
        
        results[metric_name] = {
            'point_estimate': round(point_estimates[metric_name], 4),
            'ci_lower': round(lo, 4),
            'ci_upper': round(hi, 4),
            'ci_width': round(hi - lo, 4),
            'std': round(np.std(values), 4),
        }
    
    return results


def format_table(results, label=""):
    """Format results as a publication-ready ASCII table."""
    header = f"{'Metric':<15} {'Point Est.':>10} {'95% CI':>22} {'CI Width':>10}"
    sep = "-" * 60
    rows = [f"\n{label}", sep, header, sep]
    
    for metric, vals in results.items():
        ci_str = f"[{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}]"
        rows.append(f"{metric:<15} {vals['point_estimate']:>10.4f} {ci_str:>22} {vals['ci_width']:>10.4f}")
    
    rows.append(sep)
    return "\n".join(rows)


def main():
    # Try multiple possible paths for the real-world eval predictions
    candidate_paths = [
        PROJECT_DIR / "reports" / "real_world" / "paper_eval_predictions.jsonl",
        PROJECT_DIR / "reports" / "eval_predictions.jsonl",
    ]
    
    eval_path = None
    for p in candidate_paths:
        if p.exists():
            eval_path = p
            break
    
    if eval_path is None:
        print("ERROR: Could not find evaluation predictions file.", file=sys.stderr)
        print("Looked in:", file=sys.stderr)
        for p in candidate_paths:
            print(f"  - {p}", file=sys.stderr)
        return 1
    
    print(f"Loading predictions from: {eval_path}")
    y_true, y_pred = load_predictions(eval_path)
    
    if len(y_true) == 0:
        print("ERROR: No valid predictions found in file.", file=sys.stderr)
        return 1
    
    print(f"Loaded {len(y_true)} predictions ({sum(y_true)} AKI / {len(y_true) - sum(y_true)} Normal)")
    print(f"Running {N_BOOTSTRAP:,} bootstrap iterations...")
    
    results = bootstrap_ci(y_true, y_pred)
    
    # Print formatted table
    print(format_table(results, label="Gemma-4 + TimesFM Agentic Sentinel (Real-World Holdout)"))
    
    # Save results
    out_dir = PROJECT_DIR / "reports" / "real_world"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_confidence_intervals.json"
    
    output = {
        'source_file': str(eval_path),
        'n_predictions': len(y_true),
        'n_bootstrap': N_BOOTSTRAP,
        'confidence_level': CONFIDENCE_LEVEL,
        'metrics': results,
    }
    
    with open(out_path, 'w', encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
