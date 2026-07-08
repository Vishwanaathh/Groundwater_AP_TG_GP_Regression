"""
Permutation importance for GP
=================================

GP is a kernel method, not a tree -- it has no built-in "feature importance"
the way Random Forest does. This script computes MODEL-AGNOSTIC permutation
importance instead: for each feature, its values are shuffled (breaking any
real relationship with the target), and the resulting drop in GP's test
performance measures how much that feature actually matters. A feature that
causes a big performance drop when scrambled is one GP actually relies on;
a feature that causes little to no drop isn't contributing much.

This directly answers a question central to this study (systematic variable
selection) that was previously only answered for Random Forest, not for the
actual model this study is about.

Requirements:
  pip install scikit-learn pandas numpy joblib matplotlib

Input:
  ../data/gp_test_split.csv
  ../AI/GP/gp_model.pkl, gp_scaler.pkl, gp_feature_cols.pkl

Output:
  ../data/gp_permutation_importance.csv
  ../data/plots/gp_permutation_importance.png
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

print("Script started.", flush=True)
SCRIPT_START = time.time()


def elapsed():
    return time.time() - SCRIPT_START


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
TEST_SPLIT_FILE = os.path.join(DATA_DIR, "gp_test_split.csv")

GP_DIR = "../AI/GP"
GP_MODEL_FILE = os.path.join(GP_DIR, "gp_model.pkl")
SCALER_FILE = os.path.join(GP_DIR, "gp_scaler.pkl")
FEATURE_COLS_FILE = os.path.join(GP_DIR, "gp_feature_cols.pkl")

RESULTS_OUT = os.path.join(DATA_DIR, "gp_permutation_importance.csv")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

N_REPEATS = 10  # how many times each feature is shuffled -- more repeats
                # gives a more stable estimate, at the cost of more compute
                # (each repeat requires a full GP prediction pass, which is
                # not free -- see runtime note below)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# STEP 1: Load model, scaler, features, and test data
# ---------------------------------------------------------------------------
for f in [TEST_SPLIT_FILE, GP_MODEL_FILE, SCALER_FILE, FEATURE_COLS_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"'{f}' not found. Run build_gp_model.py first.")

print(f"Loading model, scaler, features, and test data... [t={elapsed():.1f}s]", flush=True)
gp_model = joblib.load(GP_MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
feature_cols = joblib.load(FEATURE_COLS_FILE)
test_df = pd.read_csv(TEST_SPLIT_FILE, low_memory=False)

X_test = scaler.transform(test_df[feature_cols].values.astype(float))
y_test = test_df["gwl"].values.astype(float)
print(f"Test rows: {len(y_test)}, Features: {len(feature_cols)} [t={elapsed():.1f}s]", flush=True)

# ---------------------------------------------------------------------------
# STEP 2: Run permutation importance
# ---------------------------------------------------------------------------
# NOTE ON RUNTIME: permutation_importance calls model.predict() roughly
# (n_features x N_REPEATS) times internally. GP prediction cost is not
# trivial (though far cheaper than GP training). With ~29 features and 10
# repeats, that's ~290 prediction calls -- expect this to take a while,
# though nowhere near as long as the original GP training run.
print(f"Running permutation importance ({N_REPEATS} repeats per feature, "
      f"this will make ~{len(feature_cols) * N_REPEATS} prediction calls, "
      f"may take a while)... [t={elapsed():.1f}s]", flush=True)

result = permutation_importance(
    gp_model, X_test, y_test,
    n_repeats=N_REPEATS,
    random_state=RANDOM_SEED,
    scoring="neg_root_mean_squared_error",  # matches the RMSE metric used everywhere else in this study
)
print(f"Permutation importance finished. [t={elapsed():.1f}s]", flush=True)

# ---------------------------------------------------------------------------
# STEP 3: Save and report results, sorted by importance
# ---------------------------------------------------------------------------
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)

importance_df.to_csv(RESULTS_OUT, index=False)
print(f"Saved results to '{RESULTS_OUT}'", flush=True)

print("\n=== GP Permutation Importance (top to bottom) ===", flush=True)
print("(higher = shuffling this feature hurts GP's RMSE more = GP relies on it more)\n", flush=True)
print(importance_df.to_string(index=False), flush=True)

# ---------------------------------------------------------------------------
# STEP 4: Plot
# ---------------------------------------------------------------------------
print("\nBuilding permutation importance plot...", flush=True)
plot_df = importance_df.sort_values("importance_mean", ascending=True)

fig, ax = plt.subplots(figsize=(9, max(6, len(plot_df) * 0.3)))
ax.barh(plot_df["feature"], plot_df["importance_mean"],
        xerr=plot_df["importance_std"], color="#9467bd", edgecolor="black")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Permutation Importance (drop in RMSE-based score when shuffled)")
ax.set_title("GP Feature Importance (Permutation-Based)\n"
             "Model-agnostic, since GP has no built-in importance scores")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "gp_permutation_importance.png"), dpi=150)
plt.close()
print(f"Saved '{os.path.join(PLOTS_DIR, 'gp_permutation_importance.png')}'", flush=True)

print(f"\nDone. Total runtime: {elapsed():.1f}s", flush=True)