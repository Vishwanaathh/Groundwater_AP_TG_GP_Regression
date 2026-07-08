"""
Check whether GP's uncertainty estimates are actually calibrated
====================================================================

A confidence interval is only meaningful if it's actually correct at the
claimed rate. GP regression outputs a predictive mean AND a standard
deviation -- but nothing so far in this pipeline checks whether, say, 95%
of true test values actually fall inside the model's predicted 95%
confidence interval. If only (say) 60% of true values fall inside that
band, the uncertainty estimates are poorly calibrated, and "calibrated
uncertainty" would be a false claim.

This script checks that directly, across several confidence levels, and
produces a standard reliability diagram (nominal confidence vs. actual
empirical coverage) -- the standard way to visualize calibration quality
for a probabilistic regression model.

A well-calibrated model's points should sit close to the diagonal line
(nominal = empirical). Points BELOW the diagonal mean the model is
OVERCONFIDENT (its intervals are too narrow -- true values fall outside
them more often than claimed). Points ABOVE the diagonal mean the model is
UNDERCONFIDENT (intervals are wider than necessary).

Requirements:
  pip install scikit-learn pandas numpy joblib scipy matplotlib

Input:
  ../data/gp_test_split.csv
  ../AI/GP/gp_model.pkl, gp_scaler.pkl, gp_feature_cols.pkl

Output:
  ../data/gp_calibration_results.csv   (nominal vs empirical coverage table)
  ../data/plots/gp_calibration.png     (reliability diagram)
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
SCALER_FILE = os.path.join(GP_DIR, "gp_scaler.pkl")
FEATURE_COLS_FILE = os.path.join(GP_DIR, "gp_feature_cols.pkl")

RESULTS_OUT = os.path.join(DATA_DIR, "gp_calibration_results.csv")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Confidence levels to check -- covers a good spread, including the 95%
# level that's the headline claim in the paper draft.
CONFIDENCE_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]

# ---------------------------------------------------------------------------
# STEP 1: Load GP model, scaler, features, and test data
# ---------------------------------------------------------------------------
for required_file in [TEST_SPLIT_FILE, GP_MODEL_FILE, SCALER_FILE, FEATURE_COLS_FILE]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(
            f"'{required_file}' not found. Run build_gp_model.py first."
        )

print("Loading GP model, scaler, feature list, and test data...", flush=True)
gp_model = joblib.load(GP_MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
feature_cols = joblib.load(FEATURE_COLS_FILE)
test_df = pd.read_csv(TEST_SPLIT_FILE, low_memory=False)

X_test = scaler.transform(test_df[feature_cols].values.astype(float))
y_test = test_df["gwl"].values.astype(float)
print(f"Test rows: {len(y_test)}", flush=True)

# ---------------------------------------------------------------------------
# STEP 2: Get GP's actual predictive mean AND standard deviation
# ---------------------------------------------------------------------------
print("Computing GP predictive mean and standard deviation on test set...", flush=True)
gp_mean, gp_std = gp_model.predict(X_test, return_std=True)

if np.all(gp_std == 0):
    print("WARNING: GP predictive std is exactly zero for every prediction -- "
          "something is likely wrong with the model or kernel (a real GP "
          "should virtually never report zero uncertainty everywhere). "
          "Calibration results below will be degenerate.", flush=True)

# ---------------------------------------------------------------------------
# STEP 3: For each confidence level, compute EMPIRICAL coverage and compare
#          to the NOMINAL (claimed) coverage
# ---------------------------------------------------------------------------
print("\n=== Calibration check: nominal vs empirical coverage ===", flush=True)
print("(A well-calibrated model has empirical ~= nominal at every level.)\n", flush=True)

results = []
for conf_level in CONFIDENCE_LEVELS:
    # For a given confidence level, find the z-score multiplier for a
    # two-sided normal interval (e.g. 95% confidence -> z = 1.96)
    z = stats.norm.ppf(0.5 + conf_level / 2)

    lower = gp_mean - z * gp_std
    upper = gp_mean + z * gp_std

    inside_interval = (y_test >= lower) & (y_test <= upper)
    empirical_coverage = inside_interval.mean()

    gap = empirical_coverage - conf_level
    if abs(gap) < 0.03:
        verdict = "well-calibrated (within 3 percentage points)"
    elif gap < 0:
        verdict = "OVERCONFIDENT (intervals too narrow)"
    else:
        verdict = "underconfident (intervals wider than necessary)"

    print(f"  Nominal {conf_level*100:>5.1f}%  ->  Empirical {empirical_coverage*100:>5.1f}%  "
          f"(gap: {gap*100:+.1f} pts)  -- {verdict}", flush=True)

    results.append({
        "nominal_confidence": conf_level,
        "empirical_coverage": empirical_coverage,
        "gap_percentage_points": gap * 100,
        "verdict": verdict,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_OUT, index=False)
print(f"\nSaved calibration results to '{RESULTS_OUT}'", flush=True)

# ---------------------------------------------------------------------------
# STEP 4: Reliability diagram -- the standard visual for calibration quality
# ---------------------------------------------------------------------------
print("Building calibration (reliability) diagram...", flush=True)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.plot(results_df["nominal_confidence"], results_df["empirical_coverage"],
        "o-", color="#1f77b4", markersize=8, label="GP (Rational Quadratic kernel)")

for _, row in results_df.iterrows():
    ax.annotate(f"{row['nominal_confidence']*100:.0f}%",
                (row["nominal_confidence"], row["empirical_coverage"]),
                textcoords="offset points", xytext=(8, -4), fontsize=8)

ax.set_xlabel("Nominal Confidence Level (claimed)")
ax.set_ylabel("Empirical Coverage (actual, on test set)")
ax.set_title("GP Uncertainty Calibration Check\n"
             "(points below the diagonal = overconfident; above = underconfident)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "gp_calibration.png"), dpi=150)
plt.close()
print(f"Saved '{os.path.join(PLOTS_DIR, 'gp_calibration.png')}'", flush=True)

# ---------------------------------------------------------------------------
# STEP 5: Overall summary -- is the headline 95% CI claim actually justified?
# ---------------------------------------------------------------------------
row_95 = results_df[results_df["nominal_confidence"] == 0.95].iloc[0]
print(f"\n=== Headline check: the 95% confidence interval used in the "
      f"uncertainty band plot ===", flush=True)
print(f"Claimed: 95% of true values should fall inside the predicted interval.", flush=True)
print(f"Actual:  {row_95['empirical_coverage']*100:.1f}% of true test values "
      f"fell inside it.", flush=True)
print(f"Verdict: {row_95['verdict']}", flush=True)

print("\nDone.", flush=True)