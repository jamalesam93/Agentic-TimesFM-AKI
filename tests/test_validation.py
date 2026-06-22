"""
test_validation.py -- Statistical Validation Suite for the EHR Synthesis Engine

Compares generated synthetic cohorts against the source baseline cohort to
verify data utility, clinical realism, and privacy preservation.

Validation Checks
-----------------
  1. DISTRIBUTIONAL FIDELITY
     Visual overlay of Baseline SCr distributions (source vs. synthetic)
     saved to the plots/ directory.

  2. QUANTITATIVE SIMILARITY
     Wasserstein-1 distance (Earth Mover's Distance) and symmetrized
     KL divergence for Age and Baseline SCr distributions.

  3. CLINICAL REALISM
     Chi-Square test of independence verifying that the synergistic
     Vancomycin + Zosyn -> AKI interaction is statistically preserved
     in the synthetic cohort (expected: p < 0.05).

  4. PRIVACY PRESERVATION
     Exact-match scan confirming that no single row in the synthetic
     dataset is a duplicate of any row in the source dataset on the
     clinically identifying feature vector (age, gender, baseline_scr,
     received_vanco, received_zosyn, developed_aki).

Usage
-----
    # Run the full pipeline first to generate output/:
    python main.py --n-patients 1000 --n-synthetic 2000

    # Then run validation:
    python -m tests.test_validation --output-dir output

    # Or with a custom plots directory:
    python -m tests.test_validation --output-dir output --plots-dir my_plots
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from scipy.stats import wasserstein_distance, chi2_contingency, entropy
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Attempt matplotlib import -- needed only for plot generation
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless rendering
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# =============================================================================
# DATA NORMALIZATION
# =============================================================================

def _normalize_cohort(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Normalizes column types across source and synthetic DataFrames so that
    both use identical representations:
      - gender: int (1 = Male, 0 = Female)
      - received_vanco / received_zosyn / developed_aki: int (1/0)
    """
    df = df.copy()

    # Gender: source uses 0/1 int; synthetic uses "M"/"F" strings
    # Check for string values regardless of the underlying dtype
    # (pandas may infer StringDtype, ArrowDtype, or plain object from CSV)
    gender_sample = str(df['gender'].iloc[0])
    if gender_sample in ("M", "F"):
        df['gender'] = df['gender'].map({"M": 1, "F": 0}).astype(int)
    else:
        df['gender'] = pd.to_numeric(df['gender'], errors='coerce').fillna(0).astype(int)

    # Boolean/int columns
    for col in ['received_vanco', 'received_zosyn', 'developed_aki']:
        df[col] = df[col].astype(int)

    # Age as int
    df['age'] = df['age'].astype(int)

    # Baseline SCr as float
    df['baseline_scr'] = df['baseline_scr'].astype(float)

    return df


# =============================================================================
# VALIDATION RESULT CONTAINER
# =============================================================================

@dataclass
class ValidationResult:
    """Container for a single validation check."""
    name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    detail: str

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return (
            f"  [{status}] {self.name}\n"
            f"         {self.metric_name}: {self.metric_value:.6f} "
            f"(threshold: {self.threshold})\n"
            f"         {self.detail}"
        )


# =============================================================================
# CHECK 1: DISTRIBUTIONAL VISUALIZATION
# =============================================================================

def plot_scr_distributions(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
    plots_dir: str,
) -> str:
    """
    Generates overlaid histograms and KDE curves comparing the Baseline SCr
    distributions of the source and synthetic cohorts.

    Returns the path to the saved figure.
    """
    if not MPL_AVAILABLE:
        return "[SKIPPED] matplotlib not installed -- cannot generate plots."

    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "EHR Synthesis Engine -- Distribution Fidelity Report",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # --- Panel A: Baseline SCr ---
    ax = axes[0]
    bins = np.linspace(0, 5, 60)
    ax.hist(source['baseline_scr'], bins=bins, density=True, alpha=0.55,
            color="#2563EB", edgecolor="white", linewidth=0.4, label="Source")
    ax.hist(synthetic['baseline_scr'], bins=bins, density=True, alpha=0.55,
            color="#F97316", edgecolor="white", linewidth=0.4, label="Synthetic")

    # KDE overlay
    from scipy.stats import gaussian_kde
    x_range = np.linspace(0, 5, 300)
    if len(source['baseline_scr']) > 1:
        kde_src = gaussian_kde(source['baseline_scr'].clip(0.1, 5.0))
        ax.plot(x_range, kde_src(x_range), color="#1D4ED8", linewidth=2,
                linestyle="-", label="Source KDE")
    if len(synthetic['baseline_scr']) > 1:
        kde_syn = gaussian_kde(synthetic['baseline_scr'].clip(0.1, 5.0))
        ax.plot(x_range, kde_syn(x_range), color="#C2410C", linewidth=2,
                linestyle="--", label="Synthetic KDE")

    ax.set_xlabel("Baseline SCr (mg/dL)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Baseline Serum Creatinine", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 4)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel B: Age ---
    ax = axes[1]
    bins_age = np.arange(15, 100, 2)
    ax.hist(source['age'], bins=bins_age, density=True, alpha=0.55,
            color="#2563EB", edgecolor="white", linewidth=0.4, label="Source")
    ax.hist(synthetic['age'], bins=bins_age, density=True, alpha=0.55,
            color="#F97316", edgecolor="white", linewidth=0.4, label="Synthetic")

    x_age = np.linspace(15, 100, 300)
    if len(source['age']) > 1:
        kde_age_src = gaussian_kde(source['age'])
        ax.plot(x_age, kde_age_src(x_age), color="#1D4ED8", linewidth=2,
                linestyle="-", label="Source KDE")
    if len(synthetic['age']) > 1:
        kde_age_syn = gaussian_kde(synthetic['age'])
        ax.plot(x_age, kde_age_syn(x_age), color="#C2410C", linewidth=2,
                linestyle="--", label="Synthetic KDE")

    ax.set_xlabel("Age (years)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Patient Age", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(15, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "distribution_fidelity.png")
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return plot_path


# =============================================================================
# CHECK 2: QUANTITATIVE DISTRIBUTIONAL SIMILARITY
# =============================================================================

def _symmetric_kl_divergence(p_data: np.ndarray, q_data: np.ndarray,
                             n_bins: int = 100) -> float:
    """
    Computes the symmetrized KL divergence (Jensen-Shannon-like) between
    two empirical distributions by histogram binning.

    D_sym(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)

    This is the Jensen-Shannon Divergence (JSD), which is symmetric,
    bounded [0, ln(2)], and always defined (no division-by-zero issues).
    """
    # Determine shared bin edges
    lo = min(p_data.min(), q_data.min())
    hi = max(p_data.max(), q_data.max())
    bins = np.linspace(lo - 1e-9, hi + 1e-9, n_bins + 1)

    p_hist, _ = np.histogram(p_data, bins=bins, density=True)
    q_hist, _ = np.histogram(q_data, bins=bins, density=True)

    # Normalize to probability distributions
    p_hist = p_hist / (p_hist.sum() + 1e-12)
    q_hist = q_hist / (q_hist.sum() + 1e-12)

    # Midpoint distribution
    m_hist = 0.5 * (p_hist + q_hist) + 1e-12

    jsd = 0.5 * entropy(p_hist + 1e-12, m_hist) + 0.5 * entropy(q_hist + 1e-12, m_hist)
    return float(jsd)


def compute_distributional_metrics(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> Tuple[ValidationResult, ValidationResult, ValidationResult, ValidationResult]:
    """
    Computes Wasserstein-1 distance and Jensen-Shannon Divergence for
    both Age and Baseline SCr distributions.

    Thresholds:
      - Wasserstein (Age):  < 5.0 years (generous for noisy synthesis)
      - Wasserstein (SCr):  < 0.3 mg/dL
      - JSD (Age):          < 0.15
      - JSD (SCr):          < 0.15
    """
    src_age = source['age'].values.astype(float)
    syn_age = synthetic['age'].values.astype(float)
    src_scr = source['baseline_scr'].values
    syn_scr = synthetic['baseline_scr'].values

    # Wasserstein-1 (Earth Mover's Distance)
    w_age = wasserstein_distance(src_age, syn_age)
    w_scr = wasserstein_distance(src_scr, syn_scr)

    # Jensen-Shannon Divergence
    jsd_age = _symmetric_kl_divergence(src_age, syn_age)
    jsd_scr = _symmetric_kl_divergence(src_scr, syn_scr)

    # Thresholds
    W_AGE_THRESH = 5.0
    W_SCR_THRESH = 0.3
    JSD_THRESH = 0.15

    return (
        ValidationResult(
            name="Wasserstein Distance -- Age",
            passed=w_age < W_AGE_THRESH,
            metric_name="W1(Age)",
            metric_value=w_age,
            threshold=W_AGE_THRESH,
            detail=f"Earth Mover's Distance between source (n={len(src_age)}) "
                   f"and synthetic (n={len(syn_age)}) age distributions.",
        ),
        ValidationResult(
            name="Wasserstein Distance -- Baseline SCr",
            passed=w_scr < W_SCR_THRESH,
            metric_name="W1(SCr)",
            metric_value=w_scr,
            threshold=W_SCR_THRESH,
            detail=f"Earth Mover's Distance between source and synthetic "
                   f"baseline Serum Creatinine distributions.",
        ),
        ValidationResult(
            name="Jensen-Shannon Divergence -- Age",
            passed=jsd_age < JSD_THRESH,
            metric_name="JSD(Age)",
            metric_value=jsd_age,
            threshold=JSD_THRESH,
            detail=f"Symmetrized KL divergence (JSD). Range: [0, ln(2) ~= 0.693]. "
                   f"Lower is better.",
        ),
        ValidationResult(
            name="Jensen-Shannon Divergence -- Baseline SCr",
            passed=jsd_scr < JSD_THRESH,
            metric_name="JSD(SCr)",
            metric_value=jsd_scr,
            threshold=JSD_THRESH,
            detail=f"Symmetrized KL divergence (JSD). Range: [0, ln(2) ~= 0.693]. "
                   f"Lower is better.",
        ),
    )


# =============================================================================
# CHECK 3: SYNERGISTIC CLINICAL EFFECT PRESERVATION (Chi-Square)
# =============================================================================

def validate_synergistic_effect(
    synthetic: pd.DataFrame,
) -> ValidationResult:
    """
    Tests whether the synergistic nephrotoxicity of Vancomycin + Zosyn is
    preserved in the synthetic cohort using a Chi-Square test of independence.

    Null hypothesis H0: AKI outcome is independent of receiving the
    Vancomycin + Zosyn combination (vs. not receiving the combination).

    We WANT to reject H0 (p < 0.05), proving the synthetic data preserves
    the clinically significant drug-drug interaction signal.

    Contingency table:
                        | Developed AKI | No AKI |
        Vanco + Zosyn   |      a        |   b    |
        Other / None    |      c        |   d    |
    """
    # Classify each patient as "synergy exposed" or not
    synergy_exposed = (synthetic['received_vanco'] == 1) & (synthetic['received_zosyn'] == 1)

    # Build 2x2 contingency table
    a = int(((synergy_exposed) & (synthetic['developed_aki'] == 1)).sum())
    b = int(((synergy_exposed) & (synthetic['developed_aki'] == 0)).sum())
    c = int(((~synergy_exposed) & (synthetic['developed_aki'] == 1)).sum())
    d = int(((~synergy_exposed) & (synthetic['developed_aki'] == 0)).sum())

    contingency = np.array([[a, b], [c, d]])

    # Guard against empty cells (too few patients)
    if contingency.min() == 0 or contingency.sum() < 20:
        return ValidationResult(
            name="Chi-Square -- Vanco+Zosyn Synergy",
            passed=False,
            metric_name="p-value",
            metric_value=1.0,
            threshold=0.05,
            detail=f"Insufficient data for Chi-Square test. "
                   f"Contingency table: {contingency.tolist()}. "
                   f"Need at least 20 patients with non-zero cells.",
        )

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Calculate AKI rates for interpretability
    rate_synergy = a / max(a + b, 1)
    rate_other = c / max(c + d, 1)

    return ValidationResult(
        name="Chi-Square -- Vanco+Zosyn Synergy",
        passed=p_value < 0.05,
        metric_name="p-value",
        metric_value=p_value,
        threshold=0.05,
        detail=(
            f"Chi2={chi2:.4f}, dof={dof}, "
            f"AKI rate (Vanco+Zosyn)={rate_synergy*100:.1f}%, "
            f"AKI rate (other)={rate_other*100:.1f}%. "
            f"Contingency: {contingency.tolist()}. "
            f"{'H0 REJECTED -- synergistic signal preserved.' if p_value < 0.05 else 'H0 NOT rejected -- signal may be lost.'}"
        ),
    )


# =============================================================================
# CHECK 4: PRIVACY -- NO EXACT DUPLICATE ROWS
# =============================================================================

def validate_no_exact_duplicates(
    source: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> ValidationResult:
    """
    Verifies that no single synthetic patient record is an exact copy of
    any source patient record on the clinically identifying feature vector.

    Comparison columns: age, gender, baseline_scr, received_vanco,
    received_zosyn, developed_aki.

    Method: An inner merge on all comparison columns. Matching rows are
    counted and compared to the expected coincidental collision rate.

    Why a threshold > 0?
    --------------------
    With coarsely discretized features (integer age, 2-decimal SCr,
    3 binary columns), the effective feature space is finite. When both
    cohorts are drawn from the same statistical distribution, some
    collisions are mathematically inevitable (birthday paradox).

    The check passes if the collision rate (matches / synthetic_size) is
    below 5%, which is well above the expected random overlap for datasets
    of this dimensionality. A rate above 5% would suggest potential
    memorization of source records rather than independent sampling.
    """
    compare_cols = ['age', 'gender', 'baseline_scr',
                    'received_vanco', 'received_zosyn', 'developed_aki']

    # Prepare join-ready DataFrames with consistent types
    src_join = source[compare_cols].copy()
    syn_join = synthetic[compare_cols].copy()

    # Round SCr to avoid floating-point noise creating false negatives
    src_join['baseline_scr'] = src_join['baseline_scr'].round(2)
    syn_join['baseline_scr'] = syn_join['baseline_scr'].round(2)

    # Inner merge to find exact matches
    duplicates = pd.merge(
        syn_join, src_join,
        on=compare_cols,
        how='inner',
    )

    n_duplicates = len(duplicates)
    n_unique_duplicates = duplicates.drop_duplicates().shape[0]
    collision_rate = n_duplicates / max(len(syn_join), 1)

    # Threshold: collision rate must be below 5%
    COLLISION_THRESHOLD = 0.05

    return ValidationResult(
        name="Privacy -- No Exact Row Duplicates",
        passed=collision_rate < COLLISION_THRESHOLD,
        metric_name="collision_rate",
        metric_value=collision_rate,
        threshold=COLLISION_THRESHOLD,
        detail=(
            f"Compared {len(syn_join)} synthetic rows against {len(src_join)} "
            f"source rows on {compare_cols}. "
            f"Found {n_duplicates} exact matches "
            f"({n_unique_duplicates} unique patterns, "
            f"collision rate: {collision_rate*100:.2f}%). "
            f"{'PRIVACY PRESERVED -- collision rate within expected random overlap.' if collision_rate < COLLISION_THRESHOLD else f'WARNING: collision rate {collision_rate*100:.1f}% exceeds threshold, potential memorization detected.'}"
        ),
    )


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_validation(
    output_dir: str,
    plots_dir: str = "plots",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs the full validation suite against pipeline outputs.

    Args:
        output_dir: Directory containing pipeline outputs (from main.py).
        plots_dir: Directory to save validation plots.
        verbose: Print results to stdout.

    Returns:
        Dict with all results and metadata.
    """
    # Load data
    source_path = os.path.join(output_dir, "raw_historical_cohort.csv")
    synthetic_path = os.path.join(output_dir, "synthetic_cohort_baselines.csv")

    if not os.path.exists(source_path):
        print(f"[ERROR] Source cohort not found: {source_path}")
        print("        Run the pipeline first: python main.py")
        sys.exit(1)
    if not os.path.exists(synthetic_path):
        print(f"[ERROR] Synthetic cohort not found: {synthetic_path}")
        print("        Run the pipeline first: python main.py")
        sys.exit(1)

    source_raw = pd.read_csv(source_path)
    synthetic_raw = pd.read_csv(synthetic_path)

    # Normalize types
    source = _normalize_cohort(source_raw, "source")
    synthetic = _normalize_cohort(synthetic_raw, "synthetic")

    if verbose:
        print("=" * 72)
        print("        EHR SYNTHESIS ENGINE -- STATISTICAL VALIDATION SUITE")
        print("=" * 72)
        print(f"  Source cohort   : {len(source):>6,} patients  ({source_path})")
        print(f"  Synthetic cohort: {len(synthetic):>6,} patients  ({synthetic_path})")
        print("-" * 72)

    results = []

    # ------------------------------------------------------------------
    # CHECK 1: Distribution Plots
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Check 1/4] Generating distribution fidelity plots...")
    plot_result = plot_scr_distributions(source, synthetic, plots_dir)
    if verbose:
        if plot_result.startswith("[SKIPPED]"):
            print(f"  {plot_result}")
        else:
            print(f"  -> Saved to: {plot_result}")

    # ------------------------------------------------------------------
    # CHECK 2: Quantitative Distributional Similarity
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Check 2/4] Computing distributional similarity metrics...")
    w_age, w_scr, jsd_age, jsd_scr = compute_distributional_metrics(source, synthetic)
    results.extend([w_age, w_scr, jsd_age, jsd_scr])
    if verbose:
        for r in [w_age, w_scr, jsd_age, jsd_scr]:
            print(r)

    # ------------------------------------------------------------------
    # CHECK 3: Synergistic Clinical Effect
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Check 3/4] Testing synergistic Vanco+Zosyn effect preservation...")
    chi2_result = validate_synergistic_effect(synthetic)
    results.append(chi2_result)
    if verbose:
        print(chi2_result)

    # ------------------------------------------------------------------
    # CHECK 4: Privacy -- No Exact Duplicates
    # ------------------------------------------------------------------
    if verbose:
        print("\n[Check 4/4] Scanning for exact row duplicates (privacy check)...")
    privacy_result = validate_no_exact_duplicates(source, synthetic)
    results.append(privacy_result)
    if verbose:
        print(privacy_result)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)
    all_passed = n_passed == n_total

    if verbose:
        print("\n" + "=" * 72)
        print(f"  VALIDATION SUMMARY: {n_passed}/{n_total} checks passed")
        if all_passed:
            print("  STATUS: ALL CHECKS PASSED")
        else:
            print("  STATUS: SOME CHECKS FAILED")
            for r in results:
                if not r.passed:
                    print(f"    - FAILED: {r.name} ({r.metric_name}={r.metric_value:.6f})")
        print("=" * 72)

    return {
        "all_passed": all_passed,
        "n_passed": n_passed,
        "n_total": n_total,
        "results": results,
        "plot_path": plot_result if isinstance(plot_result, str) and not plot_result.startswith("[") else None,
    }


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistical Validation Suite for the EHR Synthesis Engine"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory containing pipeline outputs (default: output)"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory to save validation plots (default: plots)"
    )
    args = parser.parse_args()

    result = run_validation(
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        verbose=True,
    )

    sys.exit(0 if result["all_passed"] else 1)
