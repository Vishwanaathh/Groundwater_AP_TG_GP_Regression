"""
Head-to-head calibration check: GP vs Bayesian Ridge
========================================================

GP is not the only model in this pipeline capable of producing uncertainty
estimates -- Bayesian Ridge Regression also supports predict(return_std=True).
This script checks BOTH models' calibration side by side, using the exact
same methodology as check_gp_calibration.py (empirical coverage at several
nominal confidence levels).

Why this matters: claiming "GP uniquely provides uncertainty" is weaker than
claiming "among the models capable of producing uncertainty at all, GP's
calibration held up under direct testing" -- the second claim is only
defensible if we've actually tested the alternative, not just assumed it.

Requirements:
  pip install scikit-learn pandas numpy joblib scipy matplotlib

Input:
  ../data/gp_test_split.csv
  ../AI/GP/gp_model.pkl, gp_scaler.pkl, gp_feature_cols.pkl
  ../AI/BayesianRidge/bayesianridge_model.pkl, bayesianridge_feature_cols.pkl

Output:
  ../data/calibration_comparison_results.csv
  ../data/plots/calibration_comparison.png
"""

import pandas as pd
import numpy as np
import joblib
import os
from scipy import stats
import matplotlib.pyplot as plt

print("Script started.", flush=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
TEST_SPLIT_FILE = os.path.join(DATA_DIR, "gp_test_split.csv")

GP_DIR = "../AI/GP"
GP_MODEL_FILE = os.path.join(GP_DIR, "gp_model.pkl")
GP_SCALER_FILE = os.path.join(GP_DIR, "gp_scaler.pkl")
GP_FEATURE_COLS_FILE = os.path.join(GP_DIR, "gp_feature_cols.pkl")

BR_DIR = "../AI/BayesianRidge"
BR_MODEL_FILE = os.path.join(BR_DIR, "bayesianridge_model.pkl")
BR_FEATURE_COLS_FILE = os.path.join(BR_DIR, "bayesianridge_feature_cols.pkl")

RESULTS_OUT = os.path.join(DATA_DIR, "calibration_comparison_results.csv")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]

# ---------------------------------------------------------------------------
# STEP 1: Load everything needed for both models
# ---------------------------------------------------------------------------
required = [TEST_SPLIT_FILE, GP_MODEL_FILE, GP_SCALER_FILE, GP_FEATURE_COLS_FILE,
            BR_MODEL_FILE, BR_FEATURE_COLS_FILE]
for f in required:
    if not os.path.exists(f):
        raise FileNotFoundError(f"'{f}' not found. Run build_gp_model.py and "
                                 f"build_ml_baselines.py first.")

print("Loading test data...", flush=True)
test_df = pd.read_csv(TEST_SPLIT_FILE, low_memory=False)
y_test = test_df["gwl"].values.astype(float)

print("Loading GP model + scaler + features...", flush=True)
gp_model = joblib.load(GP_MODEL_FILE)
gp_scaler = joblib.load(GP_SCALER_FILE)
gp_feature_cols = joblib.load(GP_FEATURE_COLS_FILE)
X_test_gp = gp_scaler.transform(test_df[gp_feature_cols].values.astype(float))

print("Loading Bayesian Ridge model + features...", flush=True)
br_model = joblib.load(BR_MODEL_FILE)
br_feature_cols = joblib.load(BR_FEATURE_COLS_FILE)
# Bayesian Ridge was trained using the SAME shared scaler as GP (see
# build_ml_baselines.py -- it loads gp_scaler.pkl, never refits it), so we
# reuse gp_scaler here too rather than loading a separate one that doesn't exist.
X_test_br = gp_scaler.transform(test_df[br_feature_cols].values.astype(float))

# ---------------------------------------------------------------------------
# STEP 2: Get predictive mean + std for BOTH models
# ---------------------------------------------------------------------------
print("Computing predictive mean and std for both models...", flush=True)
gp_mean, gp_std = gp_model.predict(X_test_gp, return_std=True)
br_mean, br_std = br_model.predict(X_test_br, return_std=True)

for name, std in [("GP", gp_std), ("BayesianRidge", br_std)]:
    if np.all(std == 0):
        print(f"WARNING: {name} predictive std is exactly zero for every "
              f"prediction -- calibration results for {name} will be degenerate.",
              flush=True)

# ---------------------------------------------------------------------------
# STEP 3: Calibration check for both, side by side
# ---------------------------------------------------------------------------
print("\n=== Calibration comparison: GP vs Bayesian Ridge ===", flush=True)
print("(A well-calibrated model has empirical ~= nominal at every level.)\n", flush=True)

results = []
for model_name, mean, std in [("GP", gp_mean, gp_std), ("BayesianRidge", br_mean, br_std)]:
    for conf_level in CONFIDENCE_LEVELS:
        z = stats.norm.ppf(0.5 + conf_level / 2)
        lower = mean - z * std
        upper = mean + z * std
        inside = (y_test >= lower) & (y_test <= upper)
        empirical = inside.mean()
        gap = empirical - conf_level

        if abs(gap) < 0.03:
            verdict = "well-calibrated"
        elif gap < 0:
            verdict = "OVERCONFIDENT"
        else:
            verdict = "underconfident"

        print(f"  {model_name:<14} nominal {conf_level*100:>5.1f}%  ->  "
              f"empirical {empirical*100:>5.1f}%  (gap: {gap*100:+.1f} pts)  -- {verdict}",
              flush=True)

        results.append({
            "model": model_name,
            "nominal_confidence": conf_level,
            "empirical_coverage": empirical,
            "gap_percentage_points": gap * 100,
            "verdict": verdict,
        })
    print()

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_OUT, index=False)
print(f"Saved comparison results to '{RESULTS_OUT}'", flush=True)

# ---------------------------------------------------------------------------
# STEP 4: Side-by-side reliability diagram
# ---------------------------------------------------------------------------
print("Building side-by-side reliability diagram...", flush=True)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

for model_name, color, marker in [("GP", "#1f77b4", "o"), ("BayesianRidge", "#d62728", "s")]:
    sub = results_df[results_df["model"] == model_name]
    ax.plot(sub["nominal_confidence"], sub["empirical_coverage"],
            marker=marker, color=color, markersize=8, label=model_name)

ax.set_xlabel("Nominal Confidence Level (claimed)")
ax.set_ylabel("Empirical Coverage (actual, on test set)")
ax.set_title("Calibration Comparison: GP vs Bayesian Ridge\n"
             "(the only two models here capable of producing uncertainty at all)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "calibration_comparison.png"), dpi=150)
plt.close()
print(f"Saved '{os.path.join(PLOTS_DIR, 'calibration_comparison.png')}'", flush=True)

# ---------------------------------------------------------------------------
# STEP 5: Plain-language headline verdict
# ---------------------------------------------------------------------------
print("\n=== Headline verdict (95% confidence level) ===", flush=True)
for model_name in ["GP", "BayesianRidge"]:
    row = results_df[(results_df["model"] == model_name) &
                      (results_df["nominal_confidence"] == 0.95)].iloc[0]
    print(f"  {model_name}: claimed 95%, actual {row['empirical_coverage']*100:.1f}% "
          f"-- {row['verdict']}", flush=True)

print("\nDone.", flush=True)