"""
Sanity check for AP/Telangana merged groundwater dataset
===========================================================

Run this AFTER build_ap_ts_dataset.py has produced AP_TS_full_dataset.csv.
Does NOT modify the data -- purely reports what it finds, so you can decide
what (if anything) needs fixing before moving on to modeling.

What it checks:
  1. Structural sanity: row/well/period counts, duplicates, date range
  2. Missingness: % missing per column, and which periods/wells are worst
  3. Plausibility ranges: flags values physically implausible for each
     variable (e.g. negative rainfall, soil moisture outside 0-100%, etc.)
  4. GRACE window divergence: confirms the four window lengths (1m/3m/6m/12m)
     actually diverge from each other as expected once enough calendar time
     has passed since GRACE's 2002 launch -- if they're IDENTICAL far into
     the dataset, that's a sign something is still wrong with the windowing.
  5. Saves any flagged rows to a separate CSV for manual inspection.
"""

import pandas as pd
import numpy as np

INPUT_FILE = "AP_TS_full_dataset.csv"
FLAGGED_OUTPUT = "sanity_check_flagged_rows.csv"

# ---------------------------------------------------------------------------
# Plausibility ranges -- adjust these if you have better domain-specific
# bounds, these are deliberately generous to catch only clear errors, not
# borderline-but-real extreme values.
# ---------------------------------------------------------------------------
PLAUSIBLE_RANGES = {
    "gwl": (-5, 400),                  # meters below ground -- CGWB context suggested up to ~393
    "soil_moisture": (0, 100),         # GLDAS 0-10cm layer, kg/m^2 -- shouldn't be negative or absurd
    "et": (-1e-4, 2e-4),               # GLDAS evapotranspiration, kg/m^2/s
    "rainfall_1m": (0, 2000),          # mm, summed over 1 month -- generous upper bound
    "rainfall_3m": (0, 3000),
    "rainfall_6m": (0, 4000),
    "rainfall_12m": (0, 6000),
    "grace_tws": (-100, 100),          # cm equivalent water thickness anomaly
}

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["measurement_month_start"] = pd.to_datetime(df["measurement_month_start"])

print("=" * 70)
print("1. STRUCTURAL SANITY")
print("=" * 70)
print(f"Total rows: {len(df)}")
print(f"Unique wells: {df['well_id'].nunique()}")
print(f"Unique periods: {df['period'].nunique()}")
print(f"Date range: {df['measurement_month_start'].min().date()} to "
      f"{df['measurement_month_start'].max().date()}")

dupes = df.duplicated(subset=["well_id", "period"]).sum()
print(f"Duplicate (well_id, period) rows: {dupes}"
      + ("  <-- SHOULD BE ZERO, investigate if not" if dupes > 0 else ""))

expected_rows = df["well_id"].nunique() * df["period"].nunique()
print(f"Expected rows if fully rectangular (wells x periods): {expected_rows}")
print(f"Actual rows: {len(df)}  "
      f"(difference of {expected_rows - len(df)} is expected -- some periods "
      f"were excluded entirely for having no real GRACE coverage)")

print()
print("=" * 70)
print("2. MISSINGNESS PER COLUMN")
print("=" * 70)
missing_pct = (df.isna().sum() / len(df) * 100).round(2).sort_values(ascending=False)
print(missing_pct[missing_pct > 0].to_string())
if (missing_pct == 0).all():
    print("No missing values in any column.")

print()
print("Periods with the most missing GRACE data (top 10):")
grace_cols = [c for c in df.columns if c.startswith("grace_tws_")]
period_grace_missing = df.groupby("period")[grace_cols].apply(lambda g: g.isna().mean().mean() * 100)
print(period_grace_missing.sort_values(ascending=False).head(10).round(1).to_string())

print()
print("=" * 70)
print("3. PLAUSIBILITY RANGE CHECKS")
print("=" * 70)
flagged_frames = []

for col, (lo, hi) in PLAUSIBLE_RANGES.items():
    matching_cols = [c for c in df.columns if c == col or c.startswith(col + "_")]
    if col in ["et", "rainfall", "grace_tws"]:
        # these have suffixed variants (et_1m, et_3m, ...) unless col itself
        # already has a suffix (rainfall_1m is listed directly above)
        matching_cols = [c for c in df.columns if c.startswith(col + "_") or c == col]
    for c in matching_cols:
        if c not in df.columns:
            continue
        bad = df[(df[c].notna()) & ((df[c] < lo) | (df[c] > hi))]
        if len(bad) > 0:
            print(f"  {c}: {len(bad)} rows outside plausible range [{lo}, {hi}] "
                  f"(actual min={df[c].min():.4g}, max={df[c].max():.4g})")
            flagged = bad.copy()
            flagged["flag_reason"] = f"{c} outside [{lo}, {hi}]"
            flagged_frames.append(flagged[["well_id", "period", c, "flag_reason"]])
        else:
            print(f"  {c}: OK (min={df[c].min():.4g}, max={df[c].max():.4g})")

print()
print("=" * 70)
print("4. GRACE WINDOW DIVERGENCE CHECK")
print("=" * 70)
print("Checking whether grace_tws_1m/3m/6m/12m are suspiciously IDENTICAL")
print("for periods far enough past the 2002 launch that they SHOULD differ:")

# Only check periods more than 2 years after GRACE launch, where all 4
# windows should be drawing from genuinely different date ranges
check_df = df[df["measurement_month_start"] > pd.Timestamp("2004-06-01")].copy()
same_across_windows = (
    (check_df["grace_tws_1m"] == check_df["grace_tws_3m"]) &
    (check_df["grace_tws_3m"] == check_df["grace_tws_6m"]) &
    (check_df["grace_tws_6m"] == check_df["grace_tws_12m"])
)
n_suspicious = same_across_windows.sum()
print(f"Rows (post-2004) where all 4 GRACE windows are IDENTICAL: {n_suspicious} "
      f"/ {len(check_df)} ({n_suspicious/len(check_df)*100:.1f}%)")
if n_suspicious / max(len(check_df), 1) > 0.5:
    print("WARNING: more than half of post-2004 rows show identical values "
          "across all window lengths -- this suggests the multi-timescale "
          "windowing may not be working as intended. Investigate before trusting results.")
else:
    print("Looks reasonable -- most rows show genuine divergence across window lengths.")

print()
print("=" * 70)
print("5. SAVING FLAGGED ROWS")
print("=" * 70)
if flagged_frames:
    flagged_all = pd.concat(flagged_frames, ignore_index=True)
    flagged_all.to_csv(FLAGGED_OUTPUT, index=False)
    print(f"Saved {len(flagged_all)} flagged row-instances to '{FLAGGED_OUTPUT}' for manual review.")
else:
    print("Nothing flagged -- no file written.")

print()
print("Sanity check complete.")