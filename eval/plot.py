"""
Plot model comparison results, with a focus on the Gaussian Process (GP) model
================================================================================

Reads the outputs of evaluate_models.py and produces:

  Generic ML comparison plots (all models shown, GP highlighted):
    1. Bar chart comparing RMSE across all models -- GP bar highlighted
    2. Bar chart comparing R^2 / NSE across all models -- GP bar highlighted
    3. Predicted-vs-actual scatter plots, split across two figures (half the
       models each) so no single figure gets overcrowded -- GP's panel is
       outlined so it's easy to spot
    4. Residual plot for GP specifically

  Domain-specific plots (relevant to groundwater level forecasting,
  all focused on GP):
    5. Time series: actual vs GP-predicted GWL over time, for a handful of
       sample wells
    6. Spatial error map: absolute GP error at each well's real lat/lon,
       colored by magnitude
    7. Seasonal error breakdown: GP residuals grouped by CGWB's four
       measurement seasons (Jan/May/Aug/Nov)
    8. GP uncertainty bands: Gaussian Process regression's actual
       predictive standard deviation plotted as a confidence interval
       around its mean prediction -- GP's real methodological selling
       point (calibrated uncertainty), which no other model here can
       produce at all

All figures are SAVED as PNG files (not just shown), since this runs as a
plain script, not in a notebook -- you can open the PNGs directly.

Requirements:
  pip install matplotlib pandas numpy joblib scikit-learn

Input:
  ../data/model_comparison_results.csv   (from evaluate_models.py)
  ../data/model_predictions.csv          (from evaluate_models.py)
  ../data/gp_test_split.csv              (from build_gp_model.py -- needed
                                           for well_id, period, lat/lon,
                                           aquifer type metadata)
  ../AI/GP/gp_model.pkl, gp_scaler.pkl, gp_feature_cols.pkl
                                          (needed to recompute GP's
                                           predictive std for the
                                           uncertainty band plot)

Output (saved into ../data/plots/):
  rmse_comparison.png
  r2_nse_comparison.png
  predicted_vs_actual_part1.png
  predicted_vs_actual_part2.png
  gp_residuals.png
  gp_timeseries_sample_wells.png
  gp_spatial_error_map.png
  gp_seasonal_error_boxplot.png
  gp_uncertainty_bands.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

print("Script started.", flush=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "../data"
RESULTS_FILE = os.path.join(DATA_DIR, "model_comparison_results.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "model_predictions.csv")
TEST_SPLIT_FILE = os.path.join(DATA_DIR, "gp_test_split.csv")

GP_DIR = "../AI/GP"
GP_MODEL_FILE = os.path.join(GP_DIR, "gp_model.pkl")
SCALER_FILE = os.path.join(GP_DIR, "gp_scaler.pkl")
FEATURE_COLS_FILE = os.path.join(GP_DIR, "gp_feature_cols.pkl")

PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

N_SAMPLE_WELLS = 6  # how many wells to show in the time series plot
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Candidate names for the GP column/model, in case naming differs slightly
# across upstream scripts (e.g. "GP" vs "GaussianProcess" vs "Gaussian Process").
GP_NAME_CANDIDATES = ["GP", "Gaussian Process", "GaussianProcess", "GPR"]

# Colors: GP is always highlighted in gold; Persistence (naive baseline) is
# always flagged in red; everything else is neutral blue/green.
GP_HIGHLIGHT_COLOR = "#f2a900"
BASELINE_COLOR = "#d62728"
DEFAULT_COLOR_RMSE = "#1f77b4"
DEFAULT_COLOR_R2 = "#2ca02c"

for required_file in [RESULTS_FILE, PREDICTIONS_FILE]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(
            f"'{required_file}' not found. Run evaluate_models.py first."
        )

print("Loading results and predictions...", flush=True)
results_df = pd.read_csv(RESULTS_FILE)
predictions_df = pd.read_csv(PREDICTIONS_FILE)
print(f"Loaded {len(results_df)} models' worth of results.", flush=True)

model_names = results_df["model"].tolist()

# ---------------------------------------------------------------------------
# Resolve the GP model's name as it actually appears in these two files.
# ---------------------------------------------------------------------------
available_models = set(model_names)
GP_MODEL_NAME = next((c for c in GP_NAME_CANDIDATES if c in available_models), None)

if GP_MODEL_NAME is None:
    raise ValueError(
        f"Could not find a GP model among the results. "
        f"Looked for any of {GP_NAME_CANDIDATES}, but the models present "
        f"are: {sorted(available_models)}. Update GP_NAME_CANDIDATES above "
        f"to match whatever your evaluate_models.py calls the GP model."
    )

if GP_MODEL_NAME not in predictions_df.columns:
    raise ValueError(
        f"'{GP_MODEL_NAME}' found in {RESULTS_FILE} but has no matching "
        f"prediction column in {PREDICTIONS_FILE}. Columns available: "
        f"{list(predictions_df.columns)}."
    )

print(f"Using GP model name: '{GP_MODEL_NAME}'", flush=True)
gp_row = results_df[results_df["model"] == GP_MODEL_NAME].iloc[0]

# Load test split metadata (well_id, period, lat/lon, aquifer type, etc.) and
# merge it side-by-side with predictions -- both were derived from the same
# test_df in evaluate_models.py without any reordering, so row position
# lines up exactly between the two files.
metadata_available = os.path.exists(TEST_SPLIT_FILE)
if metadata_available:
    print("Loading test split metadata for domain-specific plots...", flush=True)
    test_meta = pd.read_csv(TEST_SPLIT_FILE, low_memory=False)
    if len(test_meta) != len(predictions_df):
        print(f"  WARNING: row count mismatch (metadata={len(test_meta)}, "
              f"predictions={len(predictions_df)}) -- domain-specific plots "
              f"needing metadata will be skipped, since row alignment can't "
              f"be trusted.", flush=True)
        metadata_available = False
    else:
        test_meta["measurement_month_start"] = pd.to_datetime(test_meta["measurement_month_start"])
        combined = pd.concat([test_meta.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
else:
    print(f"  '{TEST_SPLIT_FILE}' not found -- domain-specific plots needing "
          f"well/location/time metadata will be skipped.", flush=True)

true_vals = predictions_df["true_gwl"].values
gp_pred = predictions_df[GP_MODEL_NAME].values

# ---------------------------------------------------------------------------
# PLOT 1: RMSE comparison bar chart (GP highlighted)
# ---------------------------------------------------------------------------
print("Building RMSE comparison chart...", flush=True)
fig, ax = plt.subplots(figsize=(10, 6))
colors = [
    BASELINE_COLOR if m == "Persistence"
    else GP_HIGHLIGHT_COLOR if m == GP_MODEL_NAME
    else DEFAULT_COLOR_RMSE
    for m in results_df["model"]
]
ax.barh(results_df["model"], results_df["rmse"], color=colors)
ax.set_xlabel("RMSE (lower is better)")
ax.set_title("Model Comparison: RMSE on Held-Out Test Set\n(GP highlighted in gold)")
ax.invert_yaxis()  # best (lowest RMSE, since sorted) at the top
for i, v in enumerate(results_df["rmse"]):
    ax.text(v, i, f" {v:.3f}", va="center")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rmse_comparison.png"), dpi=150)
plt.close()
print(f"  Saved '{os.path.join(PLOTS_DIR, 'rmse_comparison.png')}'", flush=True)

# ---------------------------------------------------------------------------
# PLOT 2: R^2 / NSE comparison bar chart (GP highlighted)
# ---------------------------------------------------------------------------
print("Building R2/NSE comparison chart...", flush=True)
r2_sorted = results_df.sort_values("r2", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
colors = [
    BASELINE_COLOR if m == "Persistence"
    else GP_HIGHLIGHT_COLOR if m == GP_MODEL_NAME
    else DEFAULT_COLOR_R2
    for m in r2_sorted["model"]
]
ax.barh(r2_sorted["model"], r2_sorted["r2"], color=colors)
ax.set_xlabel("R^2 / NSE (higher is better; 0 = no better than predicting the mean)")
ax.set_title("Model Comparison: R^2 / NSE on Held-Out Test Set\n(GP highlighted in gold)")
ax.axvline(0, color="black", linewidth=0.8)
ax.invert_yaxis()
for i, v in enumerate(r2_sorted["r2"]):
    ax.text(v, i, f" {v:.3f}", va="center")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "r2_nse_comparison.png"), dpi=150)
plt.close()
print(f"  Saved '{os.path.join(PLOTS_DIR, 'r2_nse_comparison.png')}'", flush=True)

# ---------------------------------------------------------------------------
# PLOT 3: Predicted vs actual, split into two figures (half the models each)
# ---------------------------------------------------------------------------
print("Building predicted-vs-actual scatter plots (split in half)...", flush=True)

lims = [
    min(true_vals.min(), predictions_df[model_names].min().min()),
    max(true_vals.max(), predictions_df[model_names].max().max()),
]

n_models = len(model_names)
half = int(np.ceil(n_models / 2))
model_groups = [model_names[:half], model_names[half:]]

for part_idx, group in enumerate(model_groups, start=1):
    if len(group) == 0:
        continue
    n_group = len(group)
    n_cols = min(3, n_group)
    n_rows = int(np.ceil(n_group / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(group):
        ax = axes[i]
        pred_vals = predictions_df[name].values
        ax.scatter(true_vals, pred_vals, alpha=0.4, s=15)
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual GWL")
        ax.set_ylabel("Predicted GWL")
        row = results_df[results_df["model"] == name].iloc[0]
        title = f"{name}\nRMSE={row['rmse']:.3f}, R2={row['r2']:.3f}"
        ax.set_title(title)
        if name == GP_MODEL_NAME:
            # Outline GP's panel so it's easy to spot at a glance.
            for spine in ax.spines.values():
                spine.set_edgecolor(GP_HIGHLIGHT_COLOR)
                spine.set_linewidth(3)
            ax.set_title(title, color=GP_HIGHLIGHT_COLOR, fontweight="bold")

    for j in range(n_group, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fname = f"predicted_vs_actual_part{part_idx}.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved '{os.path.join(PLOTS_DIR, fname)}'", flush=True)

# ---------------------------------------------------------------------------
# PLOT 4: Residual plot for GP
# ---------------------------------------------------------------------------
print("Building residual plot for GP...", flush=True)
residuals = true_vals - gp_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(gp_pred, residuals, alpha=0.4, s=15, color="#1f77b4")
ax1.axhline(0, color="black", linewidth=1)
ax1.set_xlabel("Predicted GWL")
ax1.set_ylabel("Residual (Actual - Predicted)")
ax1.set_title("Residuals vs Predicted -- GP")

ax2.hist(residuals, bins=30, color="#1f77b4", edgecolor="black")
ax2.axvline(0, color="black", linewidth=1)
ax2.set_xlabel("Residual (Actual - Predicted)")
ax2.set_ylabel("Count")
ax2.set_title("Residual Distribution -- GP")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "gp_residuals.png"), dpi=150)
plt.close()
print(f"  Saved '{os.path.join(PLOTS_DIR, 'gp_residuals.png')}'", flush=True)

# ---------------------------------------------------------------------------
# PLOT 5: Time series -- actual vs GP-predicted GWL for sample wells
# ---------------------------------------------------------------------------
if metadata_available:
    print("Building GP time series plot for sample wells...", flush=True)
    unique_wells = combined["well_id"].unique()
    n_sample = min(N_SAMPLE_WELLS, len(unique_wells))
    sample_wells = np.random.choice(unique_wells, size=n_sample, replace=False)

    n_cols = 2
    n_rows = int(np.ceil(n_sample / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, well in enumerate(sample_wells):
        ax = axes[i]
        well_data = combined[combined["well_id"] == well].sort_values("measurement_month_start")
        ax.plot(well_data["measurement_month_start"], well_data["true_gwl"],
                marker="o", label="Actual", color="black")
        ax.plot(well_data["measurement_month_start"], well_data[GP_MODEL_NAME],
                marker="s", label="Predicted (GP)", color="#1f77b4")
        ax.set_title(f"Well: {well}")
        ax.set_ylabel("GWL (m below ground)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    for j in range(n_sample, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gp_timeseries_sample_wells.png"), dpi=150)
    plt.close()
    print(f"  Saved '{os.path.join(PLOTS_DIR, 'gp_timeseries_sample_wells.png')}'", flush=True)
else:
    print("SKIPPED time series plot -- test split metadata not available.", flush=True)

# ---------------------------------------------------------------------------
# PLOT 6: Spatial error map -- absolute GP error at each well's real location
# ---------------------------------------------------------------------------
if metadata_available:
    print("Building GP spatial error map...", flush=True)
    combined["abs_error"] = np.abs(combined["true_gwl"] - combined[GP_MODEL_NAME])
    # Average absolute error per well (a well may appear multiple times in
    # the test set, once per test-period reading)
    well_errors = combined.groupby(["well_id", "Latitude", "Longitude"])["abs_error"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(well_errors["Longitude"], well_errors["Latitude"],
                     c=well_errors["abs_error"], cmap="RdYlGn_r", s=80,
                     edgecolor="black", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="Mean Absolute Error (GP)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Distribution of Prediction Error -- GP\n"
                 "(Andhra Pradesh & Telangana wells)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gp_spatial_error_map.png"), dpi=150)
    plt.close()
    print(f"  Saved '{os.path.join(PLOTS_DIR, 'gp_spatial_error_map.png')}'", flush=True)
else:
    print("SKIPPED spatial error map -- test split metadata not available.", flush=True)

# ---------------------------------------------------------------------------
# PLOT 7: Seasonal error breakdown (Jan/May/Aug/Nov) for GP
# ---------------------------------------------------------------------------
if metadata_available:
    print("Building GP seasonal error boxplot...", flush=True)
    combined["season"] = combined["period"].str.split("-").str[0]
    season_order = [s for s in ["Jan", "May", "Aug", "Nov"] if s in combined["season"].unique()]
    combined["residual"] = combined["true_gwl"] - combined[GP_MODEL_NAME]

    fig, ax = plt.subplots(figsize=(8, 6))
    season_data = [combined[combined["season"] == s]["residual"].values for s in season_order]
    try:
        ax.boxplot(season_data, tick_labels=season_order)  # matplotlib >= 3.9
    except TypeError:
        ax.boxplot(season_data, labels=season_order)  # older matplotlib fallback
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("CGWB Measurement Season")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals by Season -- GP\n"
                 "(checks whether monsoon volatility hurts prediction more than dry seasons)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gp_seasonal_error_boxplot.png"), dpi=150)
    plt.close()
    print(f"  Saved '{os.path.join(PLOTS_DIR, 'gp_seasonal_error_boxplot.png')}'", flush=True)
else:
    print("SKIPPED seasonal error boxplot -- test split metadata not available.", flush=True)

# ---------------------------------------------------------------------------
# PLOT 8: GP uncertainty bands -- GP's real predictive std, not just a point
# ---------------------------------------------------------------------------
gp_files_available = all(os.path.exists(f) for f in [GP_MODEL_FILE, SCALER_FILE, FEATURE_COLS_FILE])
if metadata_available and gp_files_available:
    print("Building GP uncertainty band plot (recomputing predictive std)...", flush=True)
    gp_model = joblib.load(GP_MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_cols = joblib.load(FEATURE_COLS_FILE)

    # Pick ONE sample well with enough test points to make a readable plot
    well_counts = combined["well_id"].value_counts()
    plot_well = well_counts.index[0]
    well_rows = combined[combined["well_id"] == plot_well].sort_values("measurement_month_start")

    X_well = scaler.transform(well_rows[feature_cols].values.astype(float))
    gp_mean, gp_std = gp_model.predict(X_well, return_std=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_axis = well_rows["measurement_month_start"]
    ax.plot(x_axis, well_rows["true_gwl"], "o-", color="black", label="Actual")
    ax.plot(x_axis, gp_mean, "s-", color="#1f77b4", label="GP Predicted Mean")
    ax.fill_between(x_axis, gp_mean - 1.96 * gp_std, gp_mean + 1.96 * gp_std,
                     color="#1f77b4", alpha=0.2, label="95% Confidence Interval")
    ax.set_ylabel("GWL (m below ground)")
    ax.set_title(f"Gaussian Process Uncertainty Bands -- Well: {plot_well}\n"
                 f"(the calibrated uncertainty no other model here can produce)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gp_uncertainty_bands.png"), dpi=150)
    plt.close()
    print(f"  Saved '{os.path.join(PLOTS_DIR, 'gp_uncertainty_bands.png')}'", flush=True)
else:
    print("SKIPPED GP uncertainty band plot -- GP model files or metadata not available.", flush=True)

print(f"\nAll plots saved to '{PLOTS_DIR}/'", flush=True)
print("Done.", flush=True)