#!/usr/bin/env python3
"""
Build a real-world evaluation holdout from MIMIC-IV Clinical Database Demo.

Joins patients, ICU stays, lab events (SCr + vancomycin troughs),
prescriptions (Vancomycin + Piperacillin-Tazobactam), and chart events
(MAP) to construct per-stay clinical trajectories in the exact same
chat-prompt format used for training.

Ground truth AKI labels are assigned using KDIGO Stage 1 criteria:
  SCr >= 1.5x baseline  =>  AKI_STAGE_1+
  Otherwise             =>  NORMAL

Usage:
  python scripts/build_mimic_holdout.py
  python scripts/build_mimic_holdout.py --mimic-dir path/to/mimic-iv --days 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# MIMIC-IV item IDs
SCR_ITEMIDS = [50912, 52546]        # Serum Creatinine (Blood, Chemistry)
VANCO_TROUGH_ITEMID = 51009         # Vancomycin level (Blood)
MAP_ITEMID = 220052                 # Arterial Blood Pressure mean


def load_tables(mimic_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all required MIMIC-IV tables."""
    print("Loading MIMIC-IV tables...")

    patients = pd.read_csv(mimic_dir / "hosp" / "patients.csv")
    print(f"  patients: {len(patients)} rows")

    admissions = pd.read_csv(
        mimic_dir / "hosp" / "admissions.csv",
        parse_dates=["admittime", "dischtime"],
    )
    print(f"  admissions: {len(admissions)} rows")

    icustays = pd.read_csv(
        mimic_dir / "icu" / "icustays.csv",
        parse_dates=["intime", "outtime"],
    )
    print(f"  icustays: {len(icustays)} rows")

    # Lab events — only load SCr + Vancomycin troughs
    labs = pd.read_csv(
        mimic_dir / "hosp" / "labevents.csv",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "value", "valueuom"],
    )
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    relevant_items = SCR_ITEMIDS + [VANCO_TROUGH_ITEMID]
    labs = labs[labs["itemid"].isin(relevant_items)].copy()
    print(f"  labevents (SCr + VancoTrough): {len(labs)} rows")

    # Prescriptions — only Vancomycin and Piperacillin-Tazobactam
    rx = pd.read_csv(
        mimic_dir / "hosp" / "prescriptions.csv",
        usecols=["subject_id", "hadm_id", "drug", "starttime", "stoptime"],
    )
    rx["starttime"] = pd.to_datetime(rx["starttime"])
    rx["stoptime"] = pd.to_datetime(rx["stoptime"])
    vanco_mask = rx["drug"].str.contains("Vancomycin", case=False, na=False)
    zosyn_mask = rx["drug"].str.contains("Piperacillin|Zosyn", case=False, na=False)
    rx = rx[vanco_mask | zosyn_mask].copy()
    rx["is_vanco"] = vanco_mask[rx.index]
    rx["is_zosyn"] = zosyn_mask[rx.index]
    print(f"  prescriptions (Vanco/Zosyn): {len(rx)} rows")

    # Chart events — only MAP
    ce = pd.read_csv(
        mimic_dir / "icu" / "chartevents.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "value"],
    )
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    ce = ce[ce["itemid"] == MAP_ITEMID].copy()
    ce["value"] = pd.to_numeric(ce["value"], errors="coerce")
    ce = ce.dropna(subset=["value"])
    print(f"  chartevents (MAP): {len(ce)} rows")

    return {
        "patients": patients,
        "admissions": admissions,
        "icustays": icustays,
        "labs": labs,
        "rx": rx,
        "chartevents": ce,
    }


def get_baseline_scr(
    labs: pd.DataFrame, subject_id: int, hadm_id: float, icu_intime: pd.Timestamp
) -> float | None:
    """
    Determine baseline SCr for a patient's ICU stay.

    Strategy (following KDIGO guidelines):
      1. First SCr within 48h before ICU admission
      2. If unavailable, the minimum SCr during the first 24h of ICU stay
      3. If still unavailable, return None (stay is excluded)
    """
    scr = labs[
        (labs["subject_id"] == subject_id)
        & (labs["itemid"].isin(SCR_ITEMIDS))
    ].copy()

    if scr.empty:
        return None

    # Parse values
    scr["scr_val"] = pd.to_numeric(scr["value"], errors="coerce")
    scr = scr.dropna(subset=["scr_val"])

    if scr.empty:
        return None

    # Strategy 1: within 48h before ICU admission
    window_start = icu_intime - timedelta(hours=48)
    pre_admit = scr[
        (scr["charttime"] >= window_start) & (scr["charttime"] <= icu_intime)
    ]
    if not pre_admit.empty:
        # Use the earliest SCr in this window
        return float(pre_admit.sort_values("charttime").iloc[0]["scr_val"])

    # Strategy 2: minimum SCr in first 24h of ICU stay
    first_24h = scr[
        (scr["charttime"] >= icu_intime)
        & (scr["charttime"] <= icu_intime + timedelta(hours=24))
    ]
    if not first_24h.empty:
        return float(first_24h["scr_val"].min())

    # Strategy 3: if the patient has any SCr during the admission at all, use minimum
    admit_scr = scr[scr["hadm_id"] == hadm_id] if not np.isnan(hadm_id) else scr
    if not admit_scr.empty:
        return float(admit_scr["scr_val"].min())

    return None


def is_drug_active(
    rx: pd.DataFrame, subject_id: int, hadm_id: float,
    day_start: pd.Timestamp, day_end: pd.Timestamp, drug_col: str
) -> bool:
    """Check if a specific drug was active during a given day window."""
    patient_rx = rx[
        (rx["subject_id"] == subject_id)
        & (rx[drug_col] == True)
    ]
    if not np.isnan(hadm_id):
        patient_rx = patient_rx[patient_rx["hadm_id"] == hadm_id]

    for _, row in patient_rx.iterrows():
        # Drug is active if its prescription period overlaps with the day window
        if row["starttime"] <= day_end and row["stoptime"] >= day_start:
            return True
    return False


def get_daily_map(
    ce: pd.DataFrame, subject_id: int, stay_id: int,
    day_start: pd.Timestamp, day_end: pd.Timestamp
) -> int | None:
    """Get the mean MAP for a given day window from chart events."""
    day_maps = ce[
        (ce["subject_id"] == subject_id)
        & (ce["stay_id"] == stay_id)
        & (ce["charttime"] >= day_start)
        & (ce["charttime"] < day_end)
    ]
    if day_maps.empty:
        return None
    return int(round(day_maps["value"].mean()))


def get_daily_scr(
    labs: pd.DataFrame, subject_id: int,
    day_start: pd.Timestamp, day_end: pd.Timestamp
) -> float | None:
    """Get the last SCr value for a given day window."""
    day_scr = labs[
        (labs["subject_id"] == subject_id)
        & (labs["itemid"].isin(SCR_ITEMIDS))
        & (labs["charttime"] >= day_start)
        & (labs["charttime"] < day_end)
    ].copy()
    day_scr["scr_val"] = pd.to_numeric(day_scr["value"], errors="coerce")
    day_scr = day_scr.dropna(subset=["scr_val"])

    if day_scr.empty:
        return None
    # Return the last measurement of the day
    return float(day_scr.sort_values("charttime").iloc[-1]["scr_val"])


def get_daily_vanco_trough(
    labs: pd.DataFrame, subject_id: int,
    day_start: pd.Timestamp, day_end: pd.Timestamp
) -> float:
    """Get the vancomycin trough level for a given day window."""
    day_vt = labs[
        (labs["subject_id"] == subject_id)
        & (labs["itemid"] == VANCO_TROUGH_ITEMID)
        & (labs["charttime"] >= day_start)
        & (labs["charttime"] < day_end)
    ].copy()
    day_vt["vt_val"] = pd.to_numeric(day_vt["value"], errors="coerce")
    day_vt = day_vt.dropna(subset=["vt_val"])

    if day_vt.empty:
        return 0.0
    return float(day_vt.sort_values("charttime").iloc[-1]["vt_val"])


def build_trajectory(
    tables: dict[str, pd.DataFrame],
    stay: pd.Series,
    patients: pd.DataFrame,
    max_days: int,
) -> dict | None:
    """
    Build a single patient trajectory from MIMIC-IV data for one ICU stay.

    Returns a dict in the same chat-prompt format as format_to_llm_jsonl(),
    or None if the stay lacks sufficient data.
    """
    subject_id = stay["subject_id"]
    hadm_id = stay["hadm_id"]
    stay_id = stay["stay_id"]
    icu_intime = stay["intime"]
    icu_outtime = stay["outtime"]
    los_days = stay["los"]

    # Get patient demographics
    pt = patients[patients["subject_id"] == subject_id]
    if pt.empty:
        return None
    pt = pt.iloc[0]
    age = pt["anchor_age"]
    gender = "M" if pt["gender"] == "M" else "F"

    # Get baseline SCr
    baseline_scr = get_baseline_scr(tables["labs"], subject_id, hadm_id, icu_intime)
    if baseline_scr is None or baseline_scr <= 0:
        return None

    # Determine number of days to simulate (capped at LOS and max_days)
    n_days = min(int(np.ceil(los_days)), max_days)
    if n_days < 2:
        return None  # Need at least 2 days for meaningful trajectory

    # Build daily trajectory
    trajectory_days = []
    last_known_scr = baseline_scr
    cumulative_vanco_trough = 0.0
    had_aki = False

    for day_num in range(1, n_days + 1):
        day_start = icu_intime + timedelta(days=day_num - 1)
        day_end = icu_intime + timedelta(days=day_num)

        # Drug exposure
        vanco_active = is_drug_active(
            tables["rx"], subject_id, hadm_id, day_start, day_end, "is_vanco"
        )
        zosyn_active = is_drug_active(
            tables["rx"], subject_id, hadm_id, day_start, day_end, "is_zosyn"
        )

        # MAP — use measured value or a default of 72 (normal MAP)
        map_val = get_daily_map(tables["chartevents"], subject_id, stay_id, day_start, day_end)
        if map_val is None:
            map_val = 72  # Default normal MAP when vitals not charted

        # SCr — use measured value or carry forward last known
        scr_val = get_daily_scr(tables["labs"], subject_id, day_start, day_end)
        if scr_val is not None:
            last_known_scr = scr_val
        scr_val = last_known_scr

        # Vancomycin trough
        vt = get_daily_vanco_trough(tables["labs"], subject_id, day_start, day_end)
        if vt > 0:
            cumulative_vanco_trough = vt  # Use actual measured trough
        elif vanco_active and cumulative_vanco_trough == 0:
            cumulative_vanco_trough = 0.0  # No trough measured yet

        # KDIGO AKI classification
        if scr_val >= baseline_scr * 1.5:
            had_aki = True

        risk_label = "NORMAL"
        if had_aki:
            risk_label = "AKI_STAGE_1+"

        trajectory_days.append({
            "day": day_num,
            "map": map_val,
            "vanco_active": vanco_active,
            "zosyn_active": zosyn_active,
            "vanco_trough": round(cumulative_vanco_trough, 1),
            "scr": round(scr_val, 2),
            "risk_state": risk_label,
        })

    # Determine final risk label from the last day
    final_risk = trajectory_days[-1]["risk_state"]

    # --- Textualize in the exact same format as textualization.py ---
    system_instruction = (
        "You are an AI-enabled clinical safety sentinel. Your task is to continuous-monitor "
        "ICU patient trajectories and predict the imminent onset of Medication-Induced Kidney Injury."
    )

    user_prompt = (
        f"Patient demographics: {age} yo, Sex: {gender}. "
        f"Baseline Serum Creatinine: {baseline_scr} mg/dL.\n"
        f"Initiating ICU clinical monitoring sequence:\n"
    )
    for rec in trajectory_days:
        user_prompt += (
            f"[Day {rec['day']}] MAP: {rec['map']} mmHg | "
            f"Meds Active: Vanc={rec['vanco_active']}, Zosyn={rec['zosyn_active']} | "
            f"Vanco Trough: {rec['vanco_trough']} ug/mL | "
            f"SCr: {rec['scr']} mg/dL\n"
        )

    assistant_response = (
        f"Clinical Synthesis: Patient has received nephrotoxic antibiotics. "
        f"Cumulative exposure combined with hemodynamic parameters indicates "
        f"risk status of the patient is currently assessed as: [{final_risk}]."
    )

    return {
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt.strip()},
            {"role": "assistant", "content": assistant_response},
        ],
        "_meta": {
            "source": "MIMIC-IV Demo 2.2",
            "subject_id": int(subject_id),
            "hadm_id": int(hadm_id),
            "stay_id": int(stay_id),
            "baseline_scr": baseline_scr,
            "final_scr": trajectory_days[-1]["scr"],
            "final_risk": final_risk,
            "los_days": float(los_days),
            "n_trajectory_days": len(trajectory_days),
            "received_vanco": any(d["vanco_active"] for d in trajectory_days),
            "received_zosyn": any(d["zosyn_active"] for d in trajectory_days),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build MIMIC-IV real-world evaluation holdout for DIKD AKI sentinel"
    )
    parser.add_argument(
        "--mimic-dir",
        type=Path,
        default=PROJECT_DIR / "mimic-iv-clinical-database-demo-2.2",
        help="Path to MIMIC-IV Clinical Database Demo directory",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Maximum number of ICU days per trajectory (default: 5)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "data" / "mimic_eval_holdout.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if not args.mimic_dir.exists():
        print(f"MIMIC-IV directory not found: {args.mimic_dir}", file=sys.stderr)
        return 1

    tables = load_tables(args.mimic_dir)
    icustays = tables["icustays"]

    print(f"\nProcessing {len(icustays)} ICU stays (max {args.days} days each)...")

    records = []
    skipped = 0
    for _, stay in icustays.iterrows():
        result = build_trajectory(tables, stay, tables["patients"], args.days)
        if result is not None:
            records.append(result)
        else:
            skipped += 1

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary statistics
    aki_count = sum(1 for r in records if r["_meta"]["final_risk"] == "AKI_STAGE_1+")
    normal_count = len(records) - aki_count
    vanco_count = sum(1 for r in records if r["_meta"]["received_vanco"])
    zosyn_count = sum(1 for r in records if r["_meta"]["received_zosyn"])
    both_count = sum(
        1 for r in records
        if r["_meta"]["received_vanco"] and r["_meta"]["received_zosyn"]
    )

    print(f"\n{'=' * 60}")
    print(f"  MIMIC-IV Real-World Holdout Summary")
    print(f"{'=' * 60}")
    print(f"  Total ICU stays processed : {len(icustays)}")
    print(f"  Skipped (insufficient data): {skipped}")
    print(f"  Valid trajectories         : {len(records)}")
    print(f"  ---")
    print(f"  AKI_STAGE_1+   : {aki_count} ({100 * aki_count / max(len(records), 1):.1f}%)")
    print(f"  NORMAL         : {normal_count} ({100 * normal_count / max(len(records), 1):.1f}%)")
    print(f"  ---")
    print(f"  Received Vancomycin        : {vanco_count}")
    print(f"  Received Zosyn             : {zosyn_count}")
    print(f"  Both (synergy cohort)      : {both_count}")
    print(f"{'=' * 60}")
    print(f"  Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
