"""
GP performance broken down by aquifer type and well type
=============================================================

This study's regional novelty claim rests on Andhra Pradesh/Telangana's
hard-rock, fractured crystalline aquifer setting -- but nothing so far
actually checks whether GP's performance DIFFERS across aquifer types or
well types. This script does that directly: splits the test set by
Aquifer Type and by Type of Well, and reports RMSE/MAE/R^2 for each group
separately, not just in aggregate.

NOTE: the saved test split file has "Aquifer Type" and "Type of Well"
already one-hot encoded (e.g. columns named "aquifer_Unconfined",
"welltype_Dug well") rather than as a single text column -- this is because
build_gp_model.py one-hot encodes them before saving. This script
reconstructs the original category labels from those one-hot columns before
grouping.

Requirements:
  pip install pandas numpy joblib matplotlib

Input:
  ../data/gp_test_split.csv
  ../data/model_predictions.csv   (for GP's actual predictions)

Output:
  ../data/performance_by_group.csv
  ../data/plots/performance_by_aquifer_type.png
  ../data/plots/performance_by_well_type.png
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Script started.", flush=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
TEST_SPLIT_FILE = os.path.join(DATA_DIR, "gp_test_split.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "model_predictions.csv")

RESULTS_OUT = os.path.join(DATA_DIR, "performance_by_group.csv")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

GP_MODEL_NAME = "GaussianProcess"
MIN_GROUP_SIZE = 5  # groups smaller than this are flagged as low-confidence
                     # (a single outlier can swing a tiny group's RMSE wildly)

# ---------------------------------------------------------------------------
# STEP 1: Load test metadata (one-hot encoded) and GP predictions
# ---------------------------------------------------------------------------
for f in [TEST_SPLIT_FILE, PREDICTIONS_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"'{f}' not found. Run build_gp_model.py and "
                                 f"evaluate_models.py first.")

print("Loading test metadata and predictions...", flush=True)
test_meta = pd.read_csv(TEST_SPLIT_FILE, low_memory=False)
predictions_df = pd.read_csv(PREDICTIONS_FILE)

if GP_MODEL_NAME not in predictions_df.columns:
    raise ValueError(f"'{GP_MODEL_NAME}' not found in predictions. "
                      f"Columns available: {list(predictions_df.columns)}")

if len(test_meta) != len(predictions_df):
    raise ValueError(f"Row count mismatch: test metadata has {len(test_meta)} rows, "
                      f"predictions has {len(predictions_df)} rows -- these must "
                      f"come from the exact same run to align correctly.")

combined = pd.concat([test_meta.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
print(f"Combined {len(combined)} rows.", flush=True)

# ---------------------------------------------------------------------------
# STEP 2: Reconstruct "Aquifer Type" and "Type of Well" from one-hot columns
# ---------------------------------------------------------------------------
print("Reconstructing category labels from one-hot encoded columns...", flush=True)

aquifer_cols = [c for c in combined.columns if c.startswith("aquifer_")]
welltype_cols = [c for c in combined.columns if c.startswith("welltype_")]

if not aquifer_cols:
    raise ValueError("No 'aquifer_*' columns found in the test split -- "
                      "check that build_gp_model.py's one-hot encoding step ran correctly.")
if not welltype_cols:
    raise ValueError("No 'welltype_*' columns found in the test split -- "
                      "check that build_gp_model.py's one-hot encoding step ran correctly.")


def reconstruct_category(row, onehot_cols, prefix):
    """Find which one-hot column is 1 for this row, strip the prefix, return the label."""
    active = [c for c in onehot_cols if row[c] == 1]
    if len(active) == 0:
        return "Unknown"
    return active[0][len(prefix):]


combined["Aquifer_Type_Label"] = combined.apply(
    lambda row: reconstruct_category(row, aquifer_cols, "aquifer_"), axis=1)
combined["Well_Type_Label"] = combined.apply(
    lambda row: reconstruct_category(row, welltype_cols, "welltype_"), axis=1)

print(f"Aquifer types found: {combined['Aquifer_Type_Label'].unique().tolist()}", flush=True)
print(f"Well types found: {combined['Well_Type_Label'].unique().tolist()}", flush=True)


# ---------------------------------------------------------------------------
# STEP 3: Compute metrics per group
# ---------------------------------------------------------------------------
def compute_group_metrics(group_df, group_col):
    rows = []
    for group_val, sub in group_df.groupby(group_col):
        y_true = sub["true_gwl"].values
        y_pred = sub[GP_MODEL_NAME].values
        n = len(sub)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        # R^2 is undefined/unstable for very small or zero-variance groups
        r2 = r2_score(y_true, y_pred) if n >= 2 and y_true.std() > 0 else np.nan
        flag = "LOW CONFIDENCE (small group)" if n < MIN_GROUP_SIZE else ""
        rows.append({
            "grouping": group_col, "group": group_val, "n": n,
            "rmse": rmse, "mae": mae, "r2": r2, "flag": flag,
        })
    return pd.DataFrame(rows)


print("\n=== GP Performance by Aquifer Type ===", flush=True)
aquifer_results = compute_group_metrics(combined, "Aquifer_Type_Label")
print(aquifer_results.to_string(index=False), flush=True)

print("\n=== GP Performance by Well Type ===", flush=True)
welltype_results = compute_group_metrics(combined, "Well_Type_Label")
print(welltype_results.to_string(index=False), flush=True)

all_results = pd.concat([aquifer_results, welltype_results], ignore_index=True)
all_results.to_csv(RESULTS_OUT, index=False)
print(f"\nSaved combined results to '{RESULTS_OUT}'", flush=True)

# ---------------------------------------------------------------------------
# STEP 4: Plots
# ---------------------------------------------------------------------------
for results_subset, group_name, fname in [
    (aquifer_results, "Aquifer Type", "performance_by_aquifer_type.png"),
    (welltype_results, "Well Type", "performance_by_well_type.png"),
]:
    print(f"Building performance-by-{group_name} plot...", flush=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sorted_df = results_subset.sort_values("rmse")
    colors = ["#d62728" if "LOW CONFIDENCE" in f else "#1f77b4" for f in sorted_df["flag"]]

    ax1.barh(sorted_df["group"].astype(str), sorted_df["rmse"], color=colors)
    ax1.set_xlabel("RMSE")
    ax1.set_title(f"GP RMSE by {group_name}\n(red = fewer than {MIN_GROUP_SIZE} test rows, low confidence)")
    for i, (v, n) in enumerate(zip(sorted_df["rmse"], sorted_df["n"])):
        ax1.text(v, i, f" n={n}", va="center", fontsize=8)

    ax2.barh(sorted_df["group"].astype(str), sorted_df["r2"], color=colors)
    ax2.set_xlabel("R^2")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title(f"GP R^2 by {group_name}")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved '{os.path.join(PLOTS_DIR, fname)}'", flush=True)

print("\nDone.", flush=True)