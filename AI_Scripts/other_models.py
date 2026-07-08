"""
Build and save classical ML baselines, on the EXACT SAME split as the GP model
=================================================================================

ONLY does: load the GP script's saved train/test split -> train each baseline
model -> save. No evaluation/metrics/plots -- that's a separate step.

Models built here are the ones actually confirmed used across the seven
Andhra Pradesh / Telangana groundwater papers reviewed:
  - Random Forest       (IWA paper; IEEE stacking paper)
  - XGBoost             (IWA paper)
  - CatBoost            (IWA paper)
  - LightGBM            (IWA paper)
  - Decision Tree       (IEEE stacking paper, base learner)
  - K-Nearest Neighbors (IEEE stacking paper, meta-model)
  - Bayesian Ridge      (IEEE stacking paper, meta-model)
  - MLP (shallow)       (MDPI paper, GRACE gap-filling)

NOT included: the Acta Geophysica paper's Bayesian Neural Network
(NAR/NARX architecture). A true replication needs a proper Bayesian deep
learning framework (e.g. PyMC, Pyro) to be faithful to what that paper
actually did -- out of scope for this batch of classical-ML baselines.
Flagging this honestly rather than silently omitting it without saying so.

IMPORTANT: this script does NOT recompute features or re-split the data.
It loads the exact train/test split and scaler already produced by
build_gp_model.py, so every model here sees IDENTICAL inputs to the GP
model -- this is what makes a later comparison actually fair.

Requirements:
  pip install scikit-learn pandas numpy joblib
  Optional (each is skipped with a clear message if not installed):
  pip install xgboost catboost lightgbm

Input:  ../data/gp_train_split.csv, ../data/gp_test_split.csv,
        ../AI/GP/gp_feature_cols.pkl, ../AI/GP/gp_scaler.pkl
        (all produced by build_gp_model.py -- run that first)
Output: One subfolder per model under ../AI/<ModelName>/, each containing
        the trained model (joblib) and a copy of the feature column list.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor

print("Script started.", flush=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
TRAIN_SPLIT_FILE = os.path.join(DATA_DIR, "gp_train_split.csv")
TEST_SPLIT_FILE = os.path.join(DATA_DIR, "gp_test_split.csv")

GP_DIR = "../AI/GP"
FEATURE_COLS_FILE = os.path.join(GP_DIR, "gp_feature_cols.pkl")
SCALER_FILE = os.path.join(GP_DIR, "gp_scaler.pkl")

BASE_MODEL_DIR = "../AI"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# STEP 1: Load the EXACT SAME split + scaler the GP model used
# ---------------------------------------------------------------------------
for required_file in [TRAIN_SPLIT_FILE, TEST_SPLIT_FILE, FEATURE_COLS_FILE, SCALER_FILE]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(
            f"'{required_file}' not found. Run build_gp_model.py first -- "
            f"this script depends on the exact split/scaler it produces."
        )

print(f"Loading split and scaler from GP model run...", flush=True)
train_df = pd.read_csv(TRAIN_SPLIT_FILE, low_memory=False)
test_df = pd.read_csv(TEST_SPLIT_FILE, low_memory=False)
feature_cols = joblib.load(FEATURE_COLS_FILE)
scaler = joblib.load(SCALER_FILE)
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}, Features: {len(feature_cols)}", flush=True)

# Apply the SAME already-fitted scaler -- do not refit it here. Refitting
# would mean these models see slightly different feature scaling than GP
# did, undermining the point of a fair, identical-inputs comparison.
X_train = scaler.transform(train_df[feature_cols].values.astype(float))
y_train = train_df["gwl"].values.astype(float)
print("Applied GP's already-fitted scaler (not refit) -- inputs are identical to what GP trained on.", flush=True)

# ---------------------------------------------------------------------------
# STEP 2: Define and train each model, saving as it goes
# ---------------------------------------------------------------------------
def save_model(model, name):
    model_dir = os.path.join(BASE_MODEL_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"{name.lower()}_model.pkl"))
    joblib.dump(feature_cols, os.path.join(model_dir, f"{name.lower()}_feature_cols.pkl"))
    print(f"  Saved '{name}' to '{model_dir}'", flush=True)


print("\nTraining Random Forest...", flush=True)
rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)
save_model(rf, "RandomForest")

print("\nTraining Decision Tree...", flush=True)
dt = DecisionTreeRegressor(random_state=RANDOM_SEED)
dt.fit(X_train, y_train)
save_model(dt, "DecisionTree")

print("\nTraining K-Nearest Neighbors...", flush=True)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
save_model(knn, "KNN")

print("\nTraining Bayesian Ridge...", flush=True)
br = BayesianRidge()
br.fit(X_train, y_train)
save_model(br, "BayesianRidge")

print("\nTraining shallow MLP (matches the MDPI paper's approach)...", flush=True)
mlp = MLPRegressor(hidden_layer_sizes=(16,), max_iter=5000, random_state=RANDOM_SEED)
mlp.fit(X_train, y_train)
save_model(mlp, "MLP")

# --- Optional models: skipped gracefully if the package isn't installed ---
print("\nTraining XGBoost...", flush=True)
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=300, random_state=RANDOM_SEED)
    xgb.fit(X_train, y_train)
    save_model(xgb, "XGBoost")
except ImportError:
    print("  SKIPPED -- xgboost not installed (pip install xgboost)", flush=True)

print("\nTraining CatBoost...", flush=True)
try:
    from catboost import CatBoostRegressor
    cb = CatBoostRegressor(iterations=300, random_state=RANDOM_SEED, verbose=False)
    cb.fit(X_train, y_train)
    save_model(cb, "CatBoost")
except ImportError:
    print("  SKIPPED -- catboost not installed (pip install catboost)", flush=True)

print("\nTraining LightGBM...", flush=True)
try:
    from lightgbm import LGBMRegressor
    lgbm = LGBMRegressor(n_estimators=300, random_state=RANDOM_SEED, verbose=-1)
    lgbm.fit(X_train, y_train)
    save_model(lgbm, "LightGBM")
except ImportError:
    print("  SKIPPED -- lightgbm not installed (pip install lightgbm)", flush=True)

print("\nDone. All available models trained and saved under '../AI/<ModelName>/'.", flush=True)
print("NOTE: Bayesian Neural Network (from the Acta Geophysica NAR/NARX paper) "
      "was not built here -- needs a dedicated Bayesian deep learning framework "
      "to replicate faithfully, out of scope for this classical-ML baseline batch.", flush=True)