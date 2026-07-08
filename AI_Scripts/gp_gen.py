"""
Build and save a Gaussian Process Regression model for AP/Telangana groundwater level
========================================================================================

ONLY does: load data -> build features -> split -> build GP -> train -> save.
No evaluation metrics/plots beyond what's needed to build the model -- that's
a separate step.

Gaussian Process Regression -- classical, non-parametric, kernel-based
Bayesian regression. NOT a neural network. Chosen specifically because:
  - It suits smaller datasets well (the opposite tradeoff from deep learning,
    which typically needs far more data to avoid overfitting).
  - It gives real uncertainty quantification (a predictive mean AND a
    variance/confidence interval per prediction) -- something confirmed
    absent across every other model used for groundwater in Andhra Pradesh
    and Telangana specifically (Bayesian Neural Networks, MLP, XGBoost,
    CatBoost, LightGBM, Random Forest, stacking ensembles, Fuzzy-AHP/GIS
    approaches -- none of these output a calibrated uncertainty estimate).
  - Confirmed via direct search: no prior study applies GP Regression to
    groundwater level OR quality prediction in Andhra Pradesh or Telangana.

---------------------------------------------------------------------------
KERNEL CHOICE -- Rational Quadratic (RQ), per literature review:
  Pan, Y., Zeng, X.K., Xu, H., Sun, Y., Wang, D., & Wu, J. (2021).
  "Evaluation of Gaussian process regression kernel functions for improving
  groundwater prediction." Journal of Hydrology, 603, 126960.
  [CORRECTED CITATION -- previously misattributed to "Zhang et al. 2021" in
  an earlier version of this comment. Verified directly: the actual author
  list is Pan, Zeng, Xu, Sun, Wang, Wu. "Zhang et al. 2021" is a DIFFERENT,
  unrelated paper (on hybrid neural networks) that got confused with this
  one -- always double-check citations pulled from memory/notes before they
  go into a manuscript.]

  This is the paper specifically about kernel choice for groundwater GPR:
  compared 9 kernels (SE, Matern, RQ, and 6 sum/product combinations) across
  3 groundwater case studies. NOTE ON THE FINDING ITSELF: only abstract/
  citing-source level access has been verified so far, not the full paper.
  What's confirmed: combined kernels (e.g. Matern+RQ, SE+RQ) performed
  particularly well, and the commonly-used SE kernel had "mediocre
  performance." Whether STANDALONE RQ specifically was the single best
  performer in isolation (vs. RQ only winning as part of a combined kernel)
  has NOT been independently confirmed from the abstract alone -- get the
  full paper text before stating this as settled fact in any manuscript.
  Used here as a standalone kernel (isotropic, scalar length_scale) rather
  than a per-feature ARD variant, since scikit-learn's RationalQuadratic
  implementation only supports a scalar length_scale (no ARD/array support)
  -- this is a direct implementation constraint, not a design choice made
  here.

Requirements:
  pip install scikit-learn pandas numpy joblib

Input:  AP_TS_full_dataset.csv (2002-2018, 251 wells, from build_ap_ts_dataset.py)
Output: gp_model.pkl          (trained GaussianProcessRegressor, via joblib)
        gp_scaler.pkl         (fitted feature scaler, needed to use the model later)
        gp_feature_cols.pkl   (exact feature column order the model expects)
        ../data/gp_train_split.csv, ../data/gp_test_split.csv (the split itself)
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel

SCRIPT_START = time.time()


def elapsed():
    """Seconds since the script started, for progress messages."""
    return time.time() - SCRIPT_START


print(f"Script started. [t={elapsed():.1f}s]", flush=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_FILE = "../data/AP_TS_full_dataset.csv"

MODEL_DIR = "../AI/GP"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_OUT = os.path.join(MODEL_DIR, "gp_model.pkl")
SCALER_OUT = os.path.join(MODEL_DIR, "gp_scaler.pkl")
FEATURE_COLS_OUT = os.path.join(MODEL_DIR, "gp_feature_cols.pkl")

DATA_DIR = "../data"
TRAIN_SPLIT_OUT = os.path.join(DATA_DIR, "gp_train_split.csv")
TEST_SPLIT_OUT = os.path.join(DATA_DIR, "gp_test_split.csv")

N_LAGS = 4
TEST_FRACTION = 0.2
RANDOM_SEED = 42

# GP training cost scales roughly with the CUBE of the number of training
# points. n_restarts_optimizer=5 means 6 total optimization attempts (1
# initial + 5 restarts), each at that same n^3 cost. This is the sklearn
# default and is used here as-is -- if this proves too slow on your machine,
# the first lever to pull is lowering this, NOT subsampling rows (see note
# below on why row count is fixed).
#
# NOTE: row subsampling was deliberately NOT added as a speed lever here,
# even though it would help (GP cost scales as n^3, so cutting rows helps a
# lot). This model needs to train on the EXACT SAME rows as the other
# baseline models (build_ml_baselines.py) for the comparison to be fair --
# quietly training GP on fewer rows than Random Forest/XGBoost/etc. would
# undermine that. If this becomes too slow, the fix is something that
# doesn't touch row count (e.g. a sparse/approximate GP method), not
# subsampling.
N_RESTARTS_OPTIMIZER = 5

np.random.seed(RANDOM_SEED)
print(f"Config loaded. [t={elapsed():.1f}s]", flush=True)

# ---------------------------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------------------------
print(f"Loading '{INPUT_FILE}'... [t={elapsed():.1f}s]", flush=True)
df = pd.read_csv(INPUT_FILE, low_memory=False)
df["measurement_month_start"] = pd.to_datetime(df["measurement_month_start"])
df = df.sort_values(["well_id", "measurement_month_start"]).reset_index(drop=True)
print(f"Loaded {df.shape[0]} rows, {df['well_id'].nunique()} wells. [t={elapsed():.1f}s]", flush=True)

# ---------------------------------------------------------------------------
# STEP 2: Build features
# ---------------------------------------------------------------------------
print(f"Building features... [t={elapsed():.1f}s]", flush=True)
static_cols = ["Latitude", "Longitude", "Well Depth"]
if "Reference_Sy" in df.columns:
    static_cols.append("Reference_Sy")

# NOTE: prefix is "welltype_", not "well_" -- "well_" would collide with the
# existing "well_id" column name and accidentally sweep it into the feature
# list (a real bug caught during testing on an earlier version of this
# pipeline).
df = pd.get_dummies(df, columns=["Type of Well", "Aquifer Type"], prefix=["welltype", "aquifer"])
categorical_cols = [c for c in df.columns if c.startswith("welltype_") or c.startswith("aquifer_")]

external_cols = [c for c in df.columns if any(
    c.startswith(p) for p in ["soil_moisture_", "et_", "rainfall_", "grace_tws_"]
)]

# GRACE is monthly-resolution satellite data -- a 1-month extraction window
# frequently doesn't overlap any real image at all (confirmed: majority of
# rows have no real grace_tws_1m data). With that much missingness, the
# column becomes an imputed constant for most rows rather than carrying real
# signal, so it's dropped. The 3/6/12-month GRACE windows are far more
# complete and capture the same underlying signal at a coarser resolution.
if "grace_tws_1m" in external_cols:
    external_cols.remove("grace_tws_1m")
    print(f"Dropped 'grace_tws_1m' from features -- majority missing due to "
          f"GRACE's monthly resolution not reliably overlapping a 1-month window. "
          f"[t={elapsed():.1f}s]", flush=True)

print(f"Building lagged features per well ({df['well_id'].nunique()} wells)... [t={elapsed():.1f}s]", flush=True)
frames = []
for well_id, g in df.groupby("well_id"):
    g = g.sort_values("measurement_month_start").reset_index(drop=True)
    for lag in range(1, N_LAGS + 1):
        g[f"gwl_lag{lag}"] = g["gwl"].shift(lag)
    frames.append(g)
full = pd.concat(frames, ignore_index=True)

lag_cols = [f"gwl_lag{i}" for i in range(1, N_LAGS + 1)]
feature_cols = lag_cols + external_cols + static_cols + categorical_cols

# ---------------------------------------------------------------------------
# Row retention: only require the ESSENTIAL columns (target + lag history)
# to be present. External/static variable gaps get imputed after the split
# below (using train statistics only), rather than dropping the whole row --
# a single missing GRACE reading shouldn't throw away an otherwise-usable row.
# ---------------------------------------------------------------------------
essential_cols = lag_cols + ["gwl"]
before_essential = len(full)
full_clean = full.dropna(subset=essential_cols).reset_index(drop=True)
print(f"Rows dropped for missing target/lag history: {before_essential - len(full_clean)} "
      f"({len(full_clean)} remain) [t={elapsed():.1f}s]", flush=True)
print(f"Feature count: {len(feature_cols)} [t={elapsed():.1f}s]", flush=True)

# ---------------------------------------------------------------------------
# STEP 2b: Train/test split -- TIME-RESPECTING, per well, not random.
# ---------------------------------------------------------------------------
print(f"Splitting into train/test (last {TEST_FRACTION*100:.0f}% of each well's "
      f"timeline held out as test)... [t={elapsed():.1f}s]", flush=True)
train_rows, test_rows = [], []
for well_id, g in full_clean.groupby("well_id"):
    g = g.sort_values("measurement_month_start")
    n_test = max(1, int(len(g) * TEST_FRACTION))
    train_rows.append(g.iloc[:-n_test])
    test_rows.append(g.iloc[-n_test:])

train_df = pd.concat(train_rows, ignore_index=True).copy()
test_df = pd.concat(test_rows, ignore_index=True).copy()
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)} [t={elapsed():.1f}s]", flush=True)

# Impute missing external/static values using TRAIN statistics only, then
# apply the SAME values to test -- avoids leaking test information into how
# imputation is computed.
impute_cols = [c for c in (external_cols + static_cols) if train_df[c].isna().any() or test_df[c].isna().any()]
if impute_cols:
    print(f"Imputing missing values (train-derived median) for {len(impute_cols)} columns: "
          f"{impute_cols} [t={elapsed():.1f}s]", flush=True)
    medians = train_df[impute_cols].median()
    train_df[impute_cols] = train_df[impute_cols].fillna(medians)
    test_df[impute_cols] = test_df[impute_cols].fillna(medians)
else:
    print(f"No missing values in external/static columns -- no imputation needed. [t={elapsed():.1f}s]", flush=True)

train_df.to_csv(TRAIN_SPLIT_OUT, index=False)
test_df.to_csv(TEST_SPLIT_OUT, index=False)
print(f"Saved train split to '{TRAIN_SPLIT_OUT}' [t={elapsed():.1f}s]", flush=True)
print(f"Saved test split to '{TEST_SPLIT_OUT}' [t={elapsed():.1f}s]", flush=True)

print(f"Using all {len(train_df)} train rows (no subsampling -- must match the "
      f"baseline models' input exactly for a fair comparison). [t={elapsed():.1f}s]", flush=True)
X_raw = train_df[feature_cols].values.astype(float)
y = train_df["gwl"].values.astype(float)

# ---------------------------------------------------------------------------
# STEP 3: Scale features
# ---------------------------------------------------------------------------
print(f"Scaling features... [t={elapsed():.1f}s]", flush=True)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f"Scaling done. [t={elapsed():.1f}s]", flush=True)

# ---------------------------------------------------------------------------
# STEP 4: Build the Gaussian Process model
# ---------------------------------------------------------------------------
print(f"Building GP model... [t={elapsed():.1f}s]", flush=True)
# Kernel: a constant scale times a Rational Quadratic (RQ) kernel, plus a
# white noise term to account for measurement noise in the groundwater
# readings themselves. RQ is used here based on Pan et al. 2021 (J. Hydrol.)
# -- see module docstring for the full, corrected citation and an honest
# note on what's actually confirmed vs. still unverified about their finding.
#
# Explicit, wide bounds are given for length_scale, alpha, and noise_level --
# during testing with an earlier RBF kernel, the default bounds caused the
# optimizer to converge right at the edge of the search range (a real
# scikit-learn ConvergenceWarning), meaning the true optimal value may have
# been outside what it was allowed to search. Widening the bounds gives the
# optimizer room to actually find the best fit rather than get stuck against
# an artificial wall. The same reasoning applies to RQ's extra `alpha`
# hyperparameter (mixture-scale parameter -- controls how much the kernel
# behaves like a scale mixture of RBFs of different length scales), so it
# gets equally wide bounds rather than being left at a narrow default.
kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) \
         * RationalQuadratic(length_scale=1.0, alpha=1.0,
                              length_scale_bounds=(1e-6, 1e6), alpha_bounds=(1e-6, 1e6)) \
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 1e3))
model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=N_RESTARTS_OPTIMIZER,   # see config comment above
    normalize_y=True,
    random_state=RANDOM_SEED,
)
print(f"Model object built (kernel: {kernel}). [t={elapsed():.1f}s]", flush=True)

# ---------------------------------------------------------------------------
# STEP 5: Train
# ---------------------------------------------------------------------------
# model.fit() is a SINGLE blocking call with zero internal progress -- unlike
# an epoch loop, sklearn gives no visibility into how far along it is. Since
# GP training cost scales roughly with n^3, this can genuinely take a while
# with nothing printed the whole time, which looks identical to the script
# being stuck. To fix that, a background thread prints a heartbeat message
# every 15 seconds for as long as fit() is still running.
print(f"Training on {X.shape[0]} rows, {X.shape[1]} features "
      f"(GP training cost scales roughly with n^3 -- may take a while, "
      f"heartbeat prints every 15s so you can see it's still working)... "
      f"[t={elapsed():.1f}s]", flush=True)

_stop_heartbeat = threading.Event()


def _heartbeat():
    while not _stop_heartbeat.wait(15):
        print(f"  ... still training, {elapsed():.0f}s elapsed so far", flush=True)


_heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
_heartbeat_thread.start()

fit_start = time.time()
model.fit(X, y)
fit_duration = time.time() - fit_start

_stop_heartbeat.set()
_heartbeat_thread.join()

print(f"Training finished in {fit_duration:.1f}s. "
      f"Final log-marginal-likelihood: {model.log_marginal_likelihood_value_:.4f} "
      f"[t={elapsed():.1f}s]", flush=True)

# ---------------------------------------------------------------------------
# STEP 6: Save the model + everything needed to use it later
# ---------------------------------------------------------------------------
print(f"Saving model, scaler, and feature list... [t={elapsed():.1f}s]", flush=True)
joblib.dump(model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
joblib.dump(feature_cols, FEATURE_COLS_OUT)

print(f"\nSaved model to '{MODEL_OUT}' [t={elapsed():.1f}s]", flush=True)
print(f"Saved fitted scaler to '{SCALER_OUT}' [t={elapsed():.1f}s]", flush=True)
print(f"Saved feature column order to '{FEATURE_COLS_OUT}' [t={elapsed():.1f}s]", flush=True)
print(f"Learned kernel: {model.kernel_} [t={elapsed():.1f}s]", flush=True)
print(f"Done. Total runtime: {elapsed():.1f}s", flush=True)