"""
GP performance during documented drought years vs. normal years
====================================================================

The Telangana GRACE gap-filling paper (Kumar et al., 2022) documented
specific drought years for this region based on negative groundwater
storage anomalies: 2003-2005, 2009, and 2015-2016. This script checks
directly whether GP's predictive performance actually degrades during
these documented drought periods compared to normal years -- a real,
region-specific test grounded in an actual prior finding for this exact
area, not a generic ML diagnostic.

Requirements:
  pip install pandas numpy matplotlib scikit-learn

Input:
  ../data/gp_test_split.csv
  ../data/model_predictions.csv

Output:
  ../data/performance_by_drought_period.csv
  ../data/plots/performance_drought_vs_normal.png
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

RESULTS_OUT = os.path.join(DATA_DIR, "performance_by_drought_period.csv")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

GP_MODEL_NAME = "GaussianProcess"

# Documented drought years for this region, per Kumar et al. (2022), based
# on negative GRACE-derived groundwater storage anomalies specifically for
# Telangana. Each drought EPISODE is tracked separately (not just lumped
# into one "drought" bucket), since they may differ in severity/cause.
DROUGHT_EPISODES = {
    "2003-2005": [2003, 2004, 2005],
    "2009": [2009],
    "2015-2016": [2015, 2016],
}
ALL_DROUGHT_YEARS = set(y for years in DROUGHT_EPISODES.values() for y in years)

MIN_GROUP_SIZE = 5

# ---------------------------------------------------------------------------
# STEP 1: Load and combine test metadata + predictions
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

# Same safety check as performance_by_group.py -- these two files MUST come
# from the same run, or row alignment can't be trusted.
if len(test_meta) != len(predictions_df):
    raise ValueError(f"Row count mismatch: test metadata has {len(test_meta)} rows, "
                      f"predictions has {len(predictions_df)} rows -- these must "
                      f"come from the exact same run to align correctly.")

combined = pd.concat([test_meta.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
combined["measurement_month_start"] = pd.to_datetime(combined["measurement_month_start"])
combined["year"] = combined["measurement_month_start"].dt.year
print(f"Combined {len(combined)} rows, spanning years "
      f"{combined['year'].min()}-{combined['year'].max()}.", flush=True)

# ---------------------------------------------------------------------------
# STEP 2: Label each row as drought or normal, and by specific episode
# ---------------------------------------------------------------------------
combined["period_type"] = combined["year"].apply(
    lambda y: "Drought" if y in ALL_DROUGHT_YEARS else "Normal")


def label_episode(year):
    for episode_name, years in DROUGHT_EPISODES.items():
        if year in years:
            return episode_name
    return "Normal"


combined["drought_episode"] = combined["year"].apply(label_episode)

print(f"\nRows by period type: {combined['period_type'].value_counts().to_dict()}", flush=True)
print(f"Rows by drought episode: {combined['drought_episode'].value_counts().to_dict()}", flush=True)


# ---------------------------------------------------------------------------
# STEP 3: Compute metrics per group
# ---------------------------------------------------------------------------
def compute_group_metrics(df, group_col):
    rows = []
    for group_val, sub in df.groupby(group_col):
        y_true = sub["true_gwl"].values
        y_pred = sub[GP_MODEL_NAME].values
        n = len(sub)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) if n >= 2 and y_true.std() > 0 else np.nan
        flag = "LOW CONFIDENCE (small group)" if n < MIN_GROUP_SIZE else ""
        rows.append({"grouping": group_col, "group": group_val, "n": n,
                      "rmse": rmse, "mae": mae, "r2": r2, "flag": flag})
    return pd.DataFrame(rows)


print("\n=== GP Performance: Drought vs Normal ===", flush=True)
drought_vs_normal = compute_group_metrics(combined, "period_type")
print(drought_vs_normal.to_string(index=False), flush=True)

print("\n=== GP Performance by Specific Drought Episode ===", flush=True)
by_episode = compute_group_metrics(combined, "drought_episode")
print(by_episode.to_string(index=False), flush=True)

all_results = pd.concat([drought_vs_normal, by_episode], ignore_index=True)
all_results.to_csv(RESULTS_OUT, index=False)
print(f"\nSaved results to '{RESULTS_OUT}'", flush=True)

# ---------------------------------------------------------------------------
# STEP 4: Statistical check -- is the drought/normal RMSE difference real?
# ---------------------------------------------------------------------------
from scipy import stats as scipy_stats

drought_errors = (combined[combined["period_type"] == "Drought"]["true_gwl"] -
                  combined[combined["period_type"] == "Drought"][GP_MODEL_NAME]) ** 2
normal_errors = (combined[combined["period_type"] == "Normal"]["true_gwl"] -
                 combined[combined["period_type"] == "Normal"][GP_MODEL_NAME]) ** 2

if len(drought_errors) >= MIN_GROUP_SIZE and len(normal_errors) >= MIN_GROUP_SIZE:
    # Mann-Whitney U test: unpaired (different rows in each group, unlike the
    # paired Wilcoxon test used for model-vs-model comparison), non-parametric.
    stat, p_value = scipy_stats.mannwhitneyu(drought_errors, normal_errors, alternative="two-sided")
    print(f"\n=== Statistical check: is the drought/normal difference real? ===", flush=True)
    print(f"Mann-Whitney U test p-value: {p_value:.4f}", flush=True)
    if p_value < 0.05:
        print("SIGNIFICANT -- GP's error distribution genuinely differs between "
              "drought and normal periods (not just noise).", flush=True)
    else:
        print("NOT significant -- not enough evidence that GP performs "
              "differently during drought vs. normal periods.", flush=True)
else:
    print("\nSkipped significance test -- one or both groups too small "
          "(fewer than 5 rows).", flush=True)

# ---------------------------------------------------------------------------
# STEP 5: Plot
# ---------------------------------------------------------------------------
print("\nBuilding drought vs normal comparison plot...", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

sorted_episode = by_episode.sort_values("rmse")
colors = ["#d62728" if g != "Normal" else "#1f77b4" for g in sorted_episode["group"]]
ax1.barh(sorted_episode["group"], sorted_episode["rmse"], color=colors)
ax1.set_xlabel("RMSE")
ax1.set_title("GP RMSE by Drought Episode\n(red = documented drought period, blue = normal)")
for i, (v, n) in enumerate(zip(sorted_episode["rmse"], sorted_episode["n"])):
    ax1.text(v, i, f" n={n}", va="center", fontsize=8)

colors2 = ["#d62728" if g == "Drought" else "#1f77b4" for g in drought_vs_normal["group"]]
ax2.bar(drought_vs_normal["group"], drought_vs_normal["rmse"], color=colors2)
ax2.set_ylabel("RMSE")
ax2.set_title("GP RMSE: Drought vs Normal (aggregated)")
for i, (v, n) in enumerate(zip(drought_vs_normal["rmse"], drought_vs_normal["n"])):
    ax2.text(i, v, f"n={n}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "performance_drought_vs_normal.png"), dpi=150)
plt.close()
print(f"Saved '{os.path.join(PLOTS_DIR, 'performance_drought_vs_normal.png')}'", flush=True)

print("\nDone.", flush=True)