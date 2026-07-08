"""
Salvage check: which periods are actually corrupted?
======================================================

The column-misalignment bug only corrupts a PERIOD's entire chunk when a
variable failed for EVERY SINGLE well that period (0/251 success) -- if only
some wells failed, the column still exists (just with some real NaNs in it),
so that period's chunk was never misaligned in the first place.

This script checks EACH period independently against the plausibility ranges
and splits your existing AP_TS_full_dataset.csv into:
  - clean_periods_salvaged.csv   -- rows from periods that pass the check,
                                     safe to keep as-is, no re-extraction needed
  - periods_to_rerun.txt          -- list of period labels that ARE corrupted
                                     and need to be re-extracted from scratch

This can save you from re-running all 92 periods if only a handful are
actually broken.
"""

import pandas as pd

INPUT_FILE = "AP_TS_full_dataset.csv"
CLEAN_OUTPUT = "clean_periods_salvaged.csv"
RERUN_LIST_OUTPUT = "periods_to_rerun.txt"

# Same plausibility ranges as sanity_check_dataset.py
PLAUSIBLE_RANGES = {
    "gwl": (-5, 400),
    "soil_moisture": (0, 100),
    "et": (-1e-4, 2e-4),
    "rainfall_1m": (0, 2000),
    "rainfall_3m": (0, 3000),
    "rainfall_6m": (0, 4000),
    "rainfall_12m": (0, 6000),
    "grace_tws": (-100, 100),
}

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Loaded {len(df)} rows across {df['period'].nunique()} periods.")

# Build the same column->range mapping logic as the sanity checker
check_cols = {}
for col, (lo, hi) in PLAUSIBLE_RANGES.items():
    matching = [c for c in df.columns if c == col or c.startswith(col + "_")]
    for c in matching:
        check_cols[c] = (lo, hi)

# For each period, check what fraction of rows violate ANY plausibility range
period_status = []
for period, group in df.groupby("period"):
    n_bad = 0
    bad_cols_this_period = []
    for c, (lo, hi) in check_cols.items():
        if c not in group.columns:
            continue
        bad_mask = (group[c].notna()) & ((group[c] < lo) | (group[c] > hi))
        if bad_mask.any():
            n_bad += bad_mask.sum()
            bad_cols_this_period.append(c)
    period_status.append({
        "period": period,
        "n_rows": len(group),
        "n_implausible_values": n_bad,
        "corrupted": n_bad > 0,
        "bad_columns": ", ".join(bad_cols_this_period) if bad_cols_this_period else ""
    })

status_df = pd.DataFrame(period_status).sort_values("period")
corrupted_periods = status_df[status_df["corrupted"]]["period"].tolist()
clean_periods = status_df[~status_df["corrupted"]]["period"].tolist()

print(f"\nClean periods (safe to keep): {len(clean_periods)} / {status_df.shape[0]}")
print(f"Corrupted periods (need re-extraction): {len(corrupted_periods)} / {status_df.shape[0]}")
print(f"\nCorrupted periods list: {corrupted_periods}")

# Save the clean subset
clean_df = df[df["period"].isin(clean_periods)]
clean_df.to_csv(CLEAN_OUTPUT, index=False)
print(f"\nSaved {len(clean_df)} rows from {len(clean_periods)} clean periods to '{CLEAN_OUTPUT}'.")

# Save the rerun list as a plain text file, one period per line
with open(RERUN_LIST_OUTPUT, "w") as f:
    for p in corrupted_periods:
        f.write(p + "\n")
print(f"Saved list of periods needing re-extraction to '{RERUN_LIST_OUTPUT}'.")

print("\nFull per-period status:")
print(status_df[["period", "n_rows", "n_implausible_values", "corrupted"]].to_string(index=False))