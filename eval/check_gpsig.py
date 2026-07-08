"""
Statistically verify how GP's errors actually compare to every other model
==============================================================================

Raw RMSE differences alone don't tell you whether a gap is REAL or just
noise. This script runs a proper PAIRED statistical test between GP and
every other model, using each model's actual per-row errors on the exact
same test rows (paired, since every model was scored on identical test
data) -- this is the same style of test used in real forecasting literature
(e.g. the Diebold-Mariano test used in the Everglades ConvLSTM paper
reviewed earlier in this project) to determine if one model is GENUINELY
more accurate than another, not just numerically slightly different.

Test used: Wilcoxon signed-rank test on paired squared errors. Chosen over
a paired t-test because it doesn't assume the error differences are
normally distributed, which is a safer assumption for real-world
groundwater residuals.

Interpretation:
  p < 0.05  -> the accuracy difference between GP and that model is
               statistically significant (unlikely to be due to chance)
  p >= 0.05 -> NOT enough evidence to say the two models differ in
               accuracy -- their performance is statistically
               indistinguishable on this test set

Requirements:
  pip install pandas numpy scipy

Input:
  ../data/model_predictions.csv   (from evaluate_models.py)

Output:
  ../data/gp_significance_results.csv
"""

import pandas as pd
import numpy as np
import os
from scipy import stats

print("Script started.", flush=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
PREDICTIONS_FILE = os.path.join(DATA_DIR, "model_predictions.csv")
RESULTS_OUT = os.path.join(DATA_DIR, "gp_significance_results.csv")

GP_MODEL_NAME = "GaussianProcess"
ALPHA = 0.05  # standard significance threshold

# ---------------------------------------------------------------------------
# STEP 1: Load predictions
# ---------------------------------------------------------------------------
if not os.path.exists(PREDICTIONS_FILE):
    raise FileNotFoundError(f"'{PREDICTIONS_FILE}' not found. Run evaluate_models.py first.")

print(f"Loading '{PREDICTIONS_FILE}'...", flush=True)
df = pd.read_csv(PREDICTIONS_FILE)

if GP_MODEL_NAME not in df.columns:
    raise ValueError(f"'{GP_MODEL_NAME}' not found in predictions file. "
                      f"Columns available: {list(df.columns)}")

true_vals = df["true_gwl"].values
gp_pred = df[GP_MODEL_NAME].values
gp_squared_errors = (true_vals - gp_pred) ** 2

other_models = [c for c in df.columns if c not in ["true_gwl", GP_MODEL_NAME]]
print(f"Comparing GP against {len(other_models)} other models: {other_models}", flush=True)

# ---------------------------------------------------------------------------
# STEP 2: Paired Wilcoxon signed-rank test, GP vs each other model
# ---------------------------------------------------------------------------
print(f"\n=== Paired significance test: GP vs each model (alpha={ALPHA}) ===", flush=True)
print("(p < 0.05 = the accuracy difference is real, not just noise;", flush=True)
print(" p >= 0.05 = the two models are statistically indistinguishable "
      "on this test set)\n", flush=True)

results = []
for model_name in other_models:
    other_pred = df[model_name].values
    other_squared_errors = (true_vals - other_pred) ** 2

    error_diff = gp_squared_errors - other_squared_errors

    # If GP and the other model produce IDENTICAL predictions everywhere,
    # the signed-rank test is undefined (all differences are zero) --
    # handle this explicitly rather than let scipy raise or return NaN.
    if np.all(error_diff == 0):
        p_value = 1.0
        statistic = np.nan
    else:
        statistic, p_value = stats.wilcoxon(gp_squared_errors, other_squared_errors)

    gp_rmse = np.sqrt(gp_squared_errors.mean())
    other_rmse = np.sqrt(other_squared_errors.mean())
    pct_diff = (gp_rmse - other_rmse) / other_rmse * 100

    significant = p_value < ALPHA
    if significant:
        direction = "GP WORSE (statistically)" if gp_rmse > other_rmse else "GP BETTER (statistically)"
    else:
        direction = "NOT statistically different"

    print(f"  GP vs {model_name:<18} RMSE diff: {pct_diff:+.2f}%   p={p_value:.4f}   -> {direction}", flush=True)

    results.append({
        "compared_to": model_name,
        "gp_rmse": gp_rmse,
        "other_rmse": other_rmse,
        "pct_rmse_difference": pct_diff,
        "p_value": p_value,
        "statistically_significant": significant,
        "verdict": direction,
    })

results_df = pd.DataFrame(results).sort_values("p_value")
results_df.to_csv(RESULTS_OUT, index=False)
print(f"\nSaved results to '{RESULTS_OUT}'", flush=True)

# ---------------------------------------------------------------------------
# STEP 3: Plain-language summary -- which specific claims are ACTUALLY backed
# ---------------------------------------------------------------------------
print("\n=== Summary: which claims does this actually support? ===", flush=True)

indistinguishable = results_df[~results_df["statistically_significant"]]["compared_to"].tolist()
gp_worse = results_df[(results_df["statistically_significant"]) &
                       (results_df["gp_rmse"] > results_df["other_rmse"])]["compared_to"].tolist()
gp_better = results_df[(results_df["statistically_significant"]) &
                        (results_df["gp_rmse"] < results_df["other_rmse"])]["compared_to"].tolist()

if indistinguishable:
    print(f"GP is statistically INDISTINGUISHABLE from: {indistinguishable}", flush=True)
    print("  -> 'GP is competitive with these models' IS a defensible claim.", flush=True)
else:
    print("GP is NOT statistically indistinguishable from any other model.", flush=True)

if gp_worse:
    print(f"GP is statistically WORSE than: {gp_worse}", flush=True)
    print("  -> Do NOT claim GP has superior or comparable accuracy to these specifically.", flush=True)

if gp_better:
    print(f"GP is statistically BETTER than: {gp_better}", flush=True)

print("\nDone.", flush=True)