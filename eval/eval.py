"""
Evaluate all trained models on the shared test split
========================================================

Loads every model built so far (GP + classical ML baselines) and evaluates
them all on the EXACT SAME test set, using the EXACT SAME scaler each was
trained with -- no retraining, no re-splitting, just prediction + scoring.

Also includes two TRADITIONAL MATHEMATICAL baselines, computed fresh here
since they're near-instant (no real "training" step needed):
  - Persistence: predict gwl_lag1 (i.e. "next quarter = same as last
    quarter"). This is the standard sanity-check baseline in hydrology
    literature -- if a sophisticated model can't beat this, it isn't
    adding real value.
  - Linear Regression: ordinary least squares, closed-form, the classical
    statistical baseline that predates machine learning as a field. Fit on
    the same scaled features as every other model.

Metrics reported: RMSE, MAE, R^2, and NSE (Nash-Sutcliffe Efficiency -- the
standard metric name used in hydrology; mathematically identical to R^2 as
computed here, reported under both names since papers in this space use
NSE specifically).

Requirements:
  pip install scikit-learn pandas numpy joblib
  (xgboost, catboost, lightgbm only if you actually trained those -- each
  is skipped gracefully if its saved model file isn't found)

Input:
  ../data/gp_test_split.csv       (shared test set, from build_gp_model.py)
  ../data/gp_train_split.csv      (shared train set -- needed to fit the
                                    traditional baselines fresh)
  ../AI/GP/gp_scaler.pkl          (the ONE scaler every model actually used)
  ../AI/GP/gp_feature_cols.pkl    (the ONE feature column order every model used)
  ../AI/<ModelName>/<modelname>_model.pkl  for each trained model

Output:
  ../data/model_comparison_results.csv   (one row per model, all metrics)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Script started.", flush=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
TRAIN_SPLIT_FILE = os.path.join(DATA_DIR, "gp_train_split.csv")
TEST_SPLIT_FILE = os.path.join(DATA_DIR, "gp_test_split.csv")
RESULTS_OUT = os.path.join(DATA_DIR, "model_comparison_results.csv")
PREDICTIONS_OUT = os.path.join(DATA_DIR, "model_predictions.csv")

GP_DIR = "../AI/GP"
FEATURE_COLS_FILE = os.path.join(GP_DIR, "gp_feature_cols.pkl")
SCALER_FILE = os.path.join(GP_DIR, "gp_scaler.pkl")

BASE_MODEL_DIR = "../AI"

# Every trained model to look for, and where its file lives. Any entry not
# found on disk is skipped gracefully (printed clearly) rather than crashing
# the whole evaluation.
MODEL_LOCATIONS = {
    "GaussianProcess": os.path.join(BASE_MODEL_DIR, "GP", "gp_model.pkl"),
    "RandomForest": os.path.join(BASE_MODEL_DIR, "RandomForest", "randomforest_model.pkl"),
    "DecisionTree": os.path.join(BASE_MODEL_DIR, "DecisionTree", "decisiontree_model.pkl"),
    "KNN": os.path.join(BASE_MODEL_DIR, "KNN", "knn_model.pkl"),
    "BayesianRidge": os.path.join(BASE_MODEL_DIR, "BayesianRidge", "bayesianridge_model.pkl"),
    "MLP": os.path.join(BASE_MODEL_DIR, "MLP", "mlp_model.pkl"),
    "XGBoost": os.path.join(BASE_MODEL_DIR, "XGBoost", "xgboost_model.pkl"),
    "CatBoost": os.path.join(BASE_MODEL_DIR, "CatBoost", "catboost_model.pkl"),
    "LightGBM": os.path.join(BASE_MODEL_DIR, "LightGBM", "lightgbm_model.pkl"),
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# STEP 1: Load the shared split, scaler, and feature list
# ---------------------------------------------------------------------------
for required_file in [TRAIN_SPLIT_FILE, TEST_SPLIT_FILE, FEATURE_COLS_FILE, SCALER_FILE]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(
            f"'{required_file}' not found. Run build_gp_model.py first -- "
            f"evaluation depends on the exact split/scaler it produces."
        )

print("Loading shared split, scaler, and feature list...", flush=True)
train_df = pd.read_csv(TRAIN_SPLIT_FILE, low_memory=False)
test_df = pd.read_csv(TEST_SPLIT_FILE, low_memory=False)
feature_cols = joblib.load(FEATURE_COLS_FILE)
scaler = joblib.load(SCALER_FILE)
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}, Features: {len(feature_cols)}", flush=True)

X_train = scaler.transform(train_df[feature_cols].values.astype(float))
y_train = train_df["gwl"].values.astype(float)
X_test = scaler.transform(test_df[feature_cols].values.astype(float))
y_test = test_df["gwl"].values.astype(float)
print("Applied the shared scaler (not refit) to both train and test features.", flush=True)


# ---------------------------------------------------------------------------
# Metric helper -- now also stores raw predictions for later plotting
# ---------------------------------------------------------------------------
predictions = {"true_gwl": y_test}  # every model's predictions get added alongside this


def compute_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # NSE (Nash-Sutcliffe Efficiency): standard hydrology metric name.
    # Mathematically identical to R^2 as computed here (1 - SS_res/SS_tot),
    # reported under both names since hydrology papers specifically use NSE.
    nse = r2
    print(f"  {name:<18} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  NSE={nse:.4f}", flush=True)
    predictions[name] = y_pred
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "nse": nse}


results = []

# ---------------------------------------------------------------------------
# STEP 2: Traditional mathematical baselines (computed fresh -- near-instant)
# ---------------------------------------------------------------------------
print("\n=== Traditional mathematical baselines ===", flush=True)

# Persistence: "next quarter = same as last quarter". The standard sanity
# floor in hydrology GWL literature -- any model that can't beat this isn't
# adding real predictive value over doing nothing.
if "gwl_lag1" in test_df.columns:
    persistence_pred = test_df["gwl_lag1"].values.astype(float)
    results.append(compute_metrics("Persistence", y_test, persistence_pred))
else:
    print("  SKIPPED Persistence -- 'gwl_lag1' column not found in test split.", flush=True)

# Linear Regression: ordinary least squares, closed-form, the classical
# statistical baseline that predates machine learning as a field. Fit fresh
# here on the exact same scaled train features every other model used.
print("  Fitting Linear Regression (fresh, on shared train features)...", flush=True)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results.append(compute_metrics("LinearRegression", y_test, lr_pred))

# ---------------------------------------------------------------------------
# STEP 3: Every trained model found on disk
# ---------------------------------------------------------------------------
print("\n=== Trained models ===", flush=True)
for name, path in MODEL_LOCATIONS.items():
    if not os.path.exists(path):
        print(f"  SKIPPED {name} -- '{path}' not found (not trained/saved yet).", flush=True)
        continue
    model = joblib.load(path)
    pred = model.predict(X_test)
    results.append(compute_metrics(name, y_test, pred))

# ---------------------------------------------------------------------------
# STEP 4: Save the comparison table AND the raw predictions (for plotting)
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
results_df.to_csv(RESULTS_OUT, index=False)

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv(PREDICTIONS_OUT, index=False)
print(f"Saved raw predictions (one column per model) to '{PREDICTIONS_OUT}'", flush=True)

print(f"\nSaved comparison table to '{RESULTS_OUT}'", flush=True)
print("\n=== Final ranking (best RMSE first) ===", flush=True)
print(results_df.to_string(index=False), flush=True)
print("\nDone.", flush=True)