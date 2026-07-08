"""
Build AP/Telangana groundwater modeling dataset
=================================================

What this script does:
  1. Loads the 251 AP + Telangana wells straight from the official
     CGWB quality-controlled file (no modification to the groundwater
     level values themselves).
  2. For each well, pulls four external variables from Google Earth
     Engine, matched to the same quarterly (Jan/May/Aug/Nov) structure
     as the groundwater readings:
       - Soil moisture      (GLDAS Noah, 0-10cm layer)
       - Evapotranspiration (GLDAS Noah)
       - Rainfall           (CHIRPS daily, summed over the window)
       - Total water storage (GRACE/GRACE-FO mascon)
  3. Merges everything into one long-format table:
       well_id, period, gwl, plus one column per variable PER window length,
       e.g. rainfall_1m, rainfall_3m, rainfall_6m, rainfall_12m,
       soil_moisture_1m, ..., et_1m, ..., grace_tws_1m, ...
  4. Saves the result as a CSV.

Requirements:
  - Run this in Google Colab (Earth Engine works natively there), or
    any environment with `earthengine-api` installed and authenticated.
  - You need a free Google Earth Engine account
    (https://signup.earthengine.google.com/) registered to a project.
  - Put "CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv" in the same
    folder as this script, or update GWL_FILE below.

Key fixes applied in this version:
  - TEMPORAL ALIGNMENT: CGWB only records a MONTH, not an exact day, for each
    reading. The original version used day 28 of the measurement month as the
    window's end date -- which meant the "antecedent window" actually leaked
    into the measurement month itself, potentially including rainfall/soil
    moisture data from AFTER the well was physically measured. Fixed: the
    window now ends on the LAST DAY OF THE MONTH BEFORE the measurement month,
    so nothing in the feature window can postdate the plausible measurement
    date.
  - SPATIAL SAMPLING: the original version sampled the single raw pixel at
    each well's exact point coordinate. Given GLDAS/CHIRPS/GRACE pixels are
    large (25km / 5.5km / 100km+ respectively), a raw point risks landing
    arbitrarily on one side of a pixel boundary, and multiple nearby wells can
    collapse onto the identical pixel. Fixed: each well is now buffered into a
    disk of BUFFER_METERS_* radius (configurable per dataset, since GLDAS and
    GRACE pixels are much bigger than CHIRPS), and the reducer averages all
    pixels intersecting that buffer instead of reading one raw pixel.
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# You already have this file locally (it's the one you've been working with
# throughout) -- just make sure it's in the same folder as this script, or
# update this path to point to wherever it actually lives on your machine.
GWL_FILE = "CGWB_filtered_wells_2000_2022.csv"
OUTPUT_FILE = "AP_TS_full_dataset.csv"

# MULTI-TIMESCALE windows, SPI/SPEI-style, instead of one fixed lookback.
# A single 3-month window can't tell a short dry spell apart from month 14
# of a sustained multi-year drought (e.g. the documented 2018-2019 Telangana
# drought) -- the model would have to infer that purely from the target's
# own history. Computing each variable at several window lengths gives it
# that signal directly. Each length produces its own separate feature
# columns downstream (e.g. rainfall_1m, rainfall_3m, rainfall_6m, rainfall_12m).
WINDOW_MONTHS_LIST = [1, 3, 6, 12]

# Native pixel resolutions, used as the reduceRegions "scale" argument
EE_SCALE_GLDAS = 27830     # ~0.25 deg
EE_SCALE_GRACE = 111320    # ~1 deg
EE_SCALE_CHIRPS = 5566     # ~0.05 deg

# Buffer radius per dataset (meters) -- larger for coarser-resolution products,
# so the averaging window is meaningful relative to native pixel size rather
# than just re-sampling the same single pixel with extra steps.
BUFFER_METERS_GLDAS = 5000     # ~5 km, meaningful relative to 25 km pixels
BUFFER_METERS_CHIRPS = 2000    # ~2 km, relative to 5.5 km pixels
BUFFER_METERS_GRACE = 25000    # ~25 km, relative to 100+ km pixels

# CGWB measurement months -> map to a representative day for date math
SEASON_MONTH = {"Jan": 1, "May": 5, "Aug": 8, "Nov": 11}

# ---------------------------------------------------------------------------
# STEP 0: Earth Engine auth
# ---------------------------------------------------------------------------
# In Colab, run this once interactively (uncomment):
# ee.Authenticate()
# ee.Initialize(project="YOUR-EE-PROJECT-ID")   # <-- replace with your project ID

try:
    ee.Initialize(project="")
except Exception:
    print("Earth Engine not initialized. Run ee.Authenticate() and "
          "ee.Initialize(project='sunlit-plasma-491217-u7') first, then re-run this script.")
    raise

# ---------------------------------------------------------------------------
# STEP 1: Load the wells + groundwater level data (unmodified, as-is)
# ---------------------------------------------------------------------------
if not os.path.exists(GWL_FILE):
    raise FileNotFoundError(
        f"'{GWL_FILE}' not found. Put it in the same folder as this script "
        f"(or update GWL_FILE above with its actual path)."
    )

df = pd.read_csv(GWL_FILE, low_memory=False)
ap_ts = df[df["State"].isin(["Andhra pradesh", "Telangana"])].copy().reset_index(drop=True)

# IMPORTANT: "Station Code" in this file is stored in truncated scientific
# notation (e.g. "1.72E+14"), which causes many DIFFERENT wells to collapse
# onto the same string -- confirmed: 251 AP/Telangana rows only have 50
# unique Station Code values. Using it as a well identifier would silently
# merge unrelated wells together. Latitude/Longitude pairs are verified
# unique across all 251 wells, so we build a proper well_id from those
# instead and keep it consistently through the rest of the pipeline.
ap_ts["well_id"] = ap_ts["Latitude"].astype(str) + "_" + ap_ts["Longitude"].astype(str)
assert ap_ts["well_id"].nunique() == len(ap_ts), "well_id is not unique -- check for duplicate coordinates"

meta_cols = ["well_id", "Station Code", "Station Name", "State", "District", "Latitude", "Longitude",
             "Type of Well", "Aquifer Type", "Well Depth"]

# Reference_Sy (specific yield) isn't present in every version of this file --
# only include it if it's actually there, rather than crashing.
if "Reference_Sy" in ap_ts.columns:
    meta_cols.append("Reference_Sy")
    print("Found 'Reference_Sy' column -- including it.")
else:
    print("WARNING: 'Reference_Sy' column not found in this file -- proceeding "
          "without it. If you need specific yield, make sure you're using the "
          "version of the CSV that includes it (check the exact file you "
          "downloaded/exported).")

season_cols = [c for c in ap_ts.columns if "-" in c and c.split("-")[0] in SEASON_MONTH]

print(f"Loaded {len(ap_ts)} wells ({ap_ts['State'].value_counts().to_dict()})")
print(f"Confirmed {ap_ts['well_id'].nunique()} unique wells via lat/lon (Station Code alone is NOT reliable here).")
print(f"Found {len(season_cols)} seasonal GWL columns: {season_cols[0]} ... {season_cols[-1]}")

# Reshape groundwater level data to long format: one row per well per period
gwl_long = ap_ts.melt(
    id_vars=meta_cols,
    value_vars=season_cols,
    var_name="period",
    value_name="gwl"
)
gwl_long["gwl"] = pd.to_numeric(gwl_long["gwl"], errors="coerce")


def period_to_date(period_str):
    """
    Convert 'Jan-05' style period string to the FIRST day of the measurement
    month. We use the first day (not day 28, as in the original version)
    because the feature window needs to end strictly BEFORE this date --
    using an end-of-month date would let the window creep into days that
    could postdate the actual (unknown, day-level) measurement date.
    """
    mon, yy = period_str.split("-")
    year = 2000 + int(yy) if int(yy) < 50 else 1900 + int(yy)
    month = SEASON_MONTH[mon]
    return datetime(year, month, 1)


gwl_long["measurement_month_start"] = gwl_long["period"].apply(period_to_date)

# ---------------------------------------------------------------------------
# STEP 2: Build buffered Earth Engine FeatureCollections of the well locations
# ---------------------------------------------------------------------------
# Each well becomes a disk (not a raw point) so the reducer averages all
# pixels intersecting that disk, rather than reading one arbitrary pixel.
# Buffer size differs per dataset since GLDAS/GRACE pixels are far larger
# than CHIRPS pixels (see BUFFER_METERS_* config above).
wells_unique = ap_ts[["well_id", "Latitude", "Longitude"]].drop_duplicates()


def build_buffered_fc(buffer_meters):
    feats = []
    for _, row in wells_unique.iterrows():
        geom = ee.Geometry.Point([row["Longitude"], row["Latitude"]]).buffer(buffer_meters)
        feats.append(ee.Feature(geom, {"well_id": row["well_id"]}))
    return ee.FeatureCollection(feats)


wells_fc_gldas = build_buffered_fc(BUFFER_METERS_GLDAS)
wells_fc_chirps = build_buffered_fc(BUFFER_METERS_CHIRPS)
wells_fc_grace = build_buffered_fc(BUFFER_METERS_GRACE)
print(f"Built 3 buffered FeatureCollections ({len(wells_unique)} wells each): "
      f"GLDAS @ {BUFFER_METERS_GLDAS}m, CHIRPS @ {BUFFER_METERS_CHIRPS}m, "
      f"GRACE @ {BUFFER_METERS_GRACE}m radius.")

# ---------------------------------------------------------------------------
# STEP 3: Helper to pull one variable for one well, for one time window
# ---------------------------------------------------------------------------
GLDAS = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
CHIRPS = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
GRACE = ee.ImageCollection("NASA/GRACE/MASS_GRIDS/MASCON")


def window_bounds(measurement_month_start, months):
    """
    Returns (ee_start, ee_end, py_start, py_end) for the antecedent window,
    where `ee_end`/`py_end` is the LAST DAY OF THE MONTH BEFORE the
    measurement month -- so the window never includes any day that could
    postdate the actual (unknown-day) measurement. `start` is `months`
    calendar months before that. Both ee.Date and plain pandas Timestamp
    versions are returned: the ee.Date versions feed filterDate(), the plain
    versions let us cheaply check data availability without a network
    round-trip.
    """
    window_end = pd.Timestamp(measurement_month_start) - pd.Timedelta(days=1)
    window_start = window_end - pd.DateOffset(months=months) + pd.Timedelta(days=1)
    ee_start = ee.Date(window_start.strftime("%Y-%m-%d"))
    ee_end = ee.Date(window_end.strftime("%Y-%m-%d"))
    return ee_start, ee_end, window_start, window_end


def extract_period_values(measurement_month_start):
    """
    For a given measurement month, compute mean soil moisture, ET, GRACE TWS,
    and summed rainfall over EACH window length in WINDOW_MONTHS_LIST (all
    windows end the day before the measurement month starts -- see
    window_bounds), for every well at once (server-side reduceRegions calls).
    Each well is sampled as a buffered disk average, not a single raw pixel
    (see BUFFER_METERS_* config).

    Returns a DataFrame with one row per well_id and columns like:
      soil_moisture_1m, soil_moisture_3m, soil_moisture_6m, soil_moisture_12m,
      et_1m, et_3m, ..., rainfall_1m, ..., grace_tws_1m, ...

    IMPORTANT: GRACE satellite data only starts ~March/April 2002, and there's
    a known ~11-month gap between GRACE and GRACE-FO (2017-2018). For any
    period/window falling in those gaps, the GRACE collection is genuinely
    empty -- that's not a bug, it's real satellite mission history. Each
    dataset is extracted in its OWN try/except below so a legitimate GRACE
    gap doesn't also wipe out perfectly good GLDAS/CHIRPS data for the same
    period. Missing values from real gaps are left as NaN, not silently
    dropped or crashed on.

    NOTE: this now makes 3 reduceRegions calls PER window length, i.e.
    len(WINDOW_MONTHS_LIST) x 3 calls per period instead of 3 -- meaningfully
    slower than the single-window version. If this becomes impractical,
    consider batching windows via Earth Engine export tasks instead of
    synchronous getInfo() calls.
    """
    rows = {}

    for months in WINDOW_MONTHS_LIST:
        suffix = f"_{months}m"
        start, end, py_start, py_end = window_bounds(measurement_month_start, months=months)

        # --- GLDAS: soil moisture + evapotranspiration ---
        try:
            gldas_window = GLDAS.filterDate(start, end)
            soil_moisture_img = gldas_window.select("SoilMoi0_10cm_inst").mean()
            et_img = gldas_window.select("Evap_tavg").mean()
            gldas_combined = soil_moisture_img.rename("soil_moisture").addBands(
                et_img.rename("et")
            )
            gldas_result = gldas_combined.reduceRegions(
                collection=wells_fc_gldas, reducer=ee.Reducer.mean(), scale=EE_SCALE_GLDAS
            )
            gldas_info = gldas_result.getInfo()["features"]
            for f in gldas_info:
                p = f["properties"]
                wid = p["well_id"]
                rows.setdefault(wid, {})
                rows[wid][f"soil_moisture{suffix}"] = p.get("soil_moisture")
                rows[wid][f"et{suffix}"] = p.get("et")
        except Exception as e:
            print(f"    GLDAS ({suffix}) unavailable for this window: {e}")

        # --- CHIRPS: summed rainfall over the window ---
        try:
            rainfall_img = CHIRPS.filterDate(start, end).select("precipitation").sum().rename("rainfall")
            rainfall_result = rainfall_img.reduceRegions(
                collection=wells_fc_chirps, reducer=ee.Reducer.mean().setOutputs(["rainfall"]), scale=EE_SCALE_CHIRPS
            )
            rainfall_info = rainfall_result.getInfo()["features"]
            for f in rainfall_info:
                p = f["properties"]
                wid = p["well_id"]
                rows.setdefault(wid, {})
                rows[wid][f"rainfall{suffix}"] = p.get("rainfall")
        except Exception as e:
            print(f"    CHIRPS ({suffix}) unavailable for this window: {e}")

        # --- GRACE: mean total water storage anomaly over the window ---
        # Skip entirely (don't even call Earth Engine) for windows we already
        # know are outside real GRACE mission coverage -- faster and cleaner
        # than attempting the call and catching a failure.
        if not grace_window_has_data(py_start, py_end):
            print(f"    GRACE ({suffix}) skipped -- window falls outside real "
                  f"GRACE/GRACE-FO coverage ({py_start.date()} to {py_end.date()}).")
            continue

        try:
            grace_window = GRACE.filterDate(start, end)
            grace_img = grace_window.select("lwe_thickness").mean().rename("grace_tws")
            grace_result = grace_img.reduceRegions(
                collection=wells_fc_grace, reducer=ee.Reducer.mean().setOutputs(["grace_tws"]), scale=EE_SCALE_GRACE
            )
            grace_info = grace_result.getInfo()["features"]
            n_null_this_call = 0
            for f in grace_info:
                p = f["properties"]
                wid = p["well_id"]
                rows.setdefault(wid, {})
                val = p.get("grace_tws")
                rows[wid][f"grace_tws{suffix}"] = val
                if val is None:
                    n_null_this_call += 1
            if n_null_this_call > 0 and n_null_this_call == len(grace_info):
                # EVERY well came back null with no exception thrown -- print
                # the raw properties dict for the first well so we can see
                # exactly what Earth Engine actually returned (e.g. wrong
                # property key, or a genuinely masked/no-data pixel).
                print(f"    GRACE ({suffix}): call succeeded but ALL {len(grace_info)} "
                      f"wells returned null. Raw properties for first well: "
                      f"{grace_info[0]['properties'] if grace_info else 'no features returned'}")
        except Exception as e:
            print(f"    GRACE ({suffix}) failed for this window: {e}")

    out = pd.DataFrame.from_dict(rows, orient="index").reset_index()
    out = out.rename(columns={"index": "well_id"})
    return out


# Known real-world GRACE data availability -- used to SKIP calls entirely
# for windows we already know can't have data, instead of attempting the
# call and catching a failure. Faster, and produces cleaner output.
GRACE_MISSION_START = pd.Timestamp("2002-04-01")
GRACE_GAP_START = pd.Timestamp("2017-07-01")   # end of original GRACE mission
GRACE_GAP_END = pd.Timestamp("2018-06-01")     # start of GRACE-FO data


def grace_window_has_data(window_start, window_end):
    """True if this window could plausibly contain any real GRACE data."""
    window_start, window_end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    if window_end < GRACE_MISSION_START:
        return False
    if window_start >= GRACE_GAP_START and window_end <= GRACE_GAP_END:
        return False
    return True


def period_has_any_grace_coverage(measurement_month_start):
    """
    True if AT LEAST ONE window length for this period could have real GRACE
    data. If this is False, the whole period will get excluded from the
    final output anyway (see Step 5), so we skip it entirely here -- no
    GLDAS or CHIRPS calls either -- rather than wasting 8 Earth Engine calls
    computing values we're just going to throw away.
    """
    for months in WINDOW_MONTHS_LIST:
        _, _, py_start, py_end = window_bounds(measurement_month_start, months=months)
        if grace_window_has_data(py_start, py_end):
            return True
    return False


# ---------------------------------------------------------------------------
# STEP 4: Loop over all unique periods, extract, and merge
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# STEP 4 + 5 combined: extract each period, merge with its GWL rows, and
# APPEND to the output CSV immediately -- instead of holding everything in
# memory and writing once at the very end. This means:
#   - you can watch the file actually grow as it runs, instead of guessing
#     whether the script is stuck
#   - a crash partway through does NOT lose everything already extracted --
#     whatever's already written to disk is safe
#   - if the file ends up empty/tiny, you'll know exactly which period it
#     died on, since the CSV only ever contains fully-completed periods
# ---------------------------------------------------------------------------
unique_periods = gwl_long[["period", "measurement_month_start"]].drop_duplicates().sort_values("measurement_month_start")
print(f"Extracting external variables for {len(unique_periods)} time periods "
      f"x {len(wells_unique)} wells. This will take a while -- each period is "
      f"one batch Earth Engine call, not one call per well.")

# RESUME-AWARE: if the output file already exists (e.g. from a run that got
# interrupted -- dropped connection, sleep, closed terminal, etc.), don't
# delete it. Instead, read which periods are already safely written, skip
# those, and only continue extracting what's still missing. This matters
# because a full re-run from scratch wastes real time re-extracting periods
# that were already correctly completed.
already_done_periods = set()
if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE, low_memory=False, usecols=["period"])
    already_done_periods = set(existing["period"].unique())
    print(f"Found existing '{OUTPUT_FILE}' with {len(already_done_periods)} periods "
          f"already completed: {sorted(already_done_periods)}")
    print("Resuming -- these will be skipped, only missing periods will be extracted.")
else:
    print(f"No existing '{OUTPUT_FILE}' found -- starting fresh.")

# CRITICAL FIX: define a FIXED, canonical column schema up front and force
# every chunk to match it exactly before writing. Without this, chunks whose
# extraction partially failed (e.g. rainfall failed for every well in some
# period) end up with a DIFFERENT SET of columns than other chunks. Since
# `to_csv(mode="a", header=False)` writes columns by POSITION, not by name,
# a chunk with a different column set/order silently shifts every value
# into the WRONG column once appended -- this is what produced physically
# impossible values (e.g. soil moisture in the thousands) in earlier runs:
# it wasn't bad data, it was misaligned columns.
variable_names = ["soil_moisture", "et", "rainfall", "grace_tws"]
variable_cols = [f"{var}_{months}m" for var in variable_names for months in WINDOW_MONTHS_LIST]
CANONICAL_COLUMNS = meta_cols + ["period", "gwl", "measurement_month_start"] + variable_cols
print(f"Fixed output schema: {len(CANONICAL_COLUMNS)} columns, order locked before extraction starts.")

# If resuming, verify the existing file's columns actually match what this
# version of the script would produce -- if they don't match (e.g. you're
# resuming a file written by an older/different version of this script),
# stop and warn rather than risk re-introducing column misalignment.
if already_done_periods:
    existing_cols = list(pd.read_csv(OUTPUT_FILE, low_memory=False, nrows=0).columns)
    if existing_cols != CANONICAL_COLUMNS:
        raise ValueError(
            f"Existing '{OUTPUT_FILE}' has different columns than this script "
            f"version expects -- resuming would risk misalignment. Existing: "
            f"{existing_cols}\nExpected: {CANONICAL_COLUMNS}\n"
            f"Either delete the file and start fresh, or reconcile the mismatch first."
        )

grace_tws_col_prefix = "grace_tws_"
header_written = bool(already_done_periods)  # if resuming, header already exists on disk
skipped_periods = []
excluded_periods = []
total_rows_written = 0

for i, row in unique_periods.iterrows():
    period_label = row["period"]
    month_start = row["measurement_month_start"]

    if period_label in already_done_periods:
        continue  # already safely written in a previous run

    if not period_has_any_grace_coverage(month_start):
        skipped_periods.append(period_label)
        print(f"  skipped entirely (no real GRACE coverage in any window): {period_label}")
        continue

    try:
        result = extract_period_values(month_start)
    except Exception as e:
        print(f"  FAILED: {period_label} -- {e}")
        time.sleep(0.2)
        continue

    result["period"] = period_label

    # Merge just this period's GWL rows with the extracted external variables
    period_gwl = gwl_long[gwl_long["period"] == period_label]
    period_final = period_gwl.merge(result, on=["well_id", "period"], how="left")

    # Force this chunk into the EXACT same column set/order as every other
    # chunk, regardless of which variables succeeded or failed extraction
    # this period. This is what prevents column misalignment on append --
    # any column genuinely missing this period becomes an explicit NaN
    # rather than shifting every subsequent column's values over by one.
    period_final = period_final.reindex(columns=CANONICAL_COLUMNS)

    # If every GRACE column for this period is entirely empty, exclude this
    # period from the output entirely rather than writing all-NaN rows.
    # IMPORTANT: this check runs AFTER the reindex above, and uses the fixed
    # canonical grace column names directly -- not columns dynamically
    # discovered from period_final. If it ran BEFORE reindex and GRACE failed
    # so completely that zero grace columns existed yet (not just NaN, but
    # literally absent), `grace_cols` would come back as an empty list, and
    # `if grace_cols and ...` would silently evaluate False -- meaning the
    # exclusion would never fire and the period would get written anyway.
    grace_cols = [c for c in CANONICAL_COLUMNS if c.startswith(grace_tws_col_prefix)]
    if period_final[grace_cols].isna().all(axis=None):
        excluded_periods.append(period_label)
        print(f"  excluded (extraction ran but produced no real GRACE data): {period_label}")
        time.sleep(0.2)
        continue

    # Append to disk right now -- header only on the very first write
    period_final.to_csv(OUTPUT_FILE, mode="a", header=not header_written, index=False)
    header_written = True
    total_rows_written += len(period_final)

    print(f"  done: {period_label}  (+{len(period_final)} rows, {total_rows_written} total so far)")
    time.sleep(0.2)  # be polite to the Earth Engine API

print(f"\nFinished. Wrote {total_rows_written} total rows to '{OUTPUT_FILE}'.")
print(f"Skipped upfront (no GRACE coverage possible): {len(skipped_periods)} periods -> {skipped_periods}")
print(f"Excluded after extraction (no real GRACE data despite passing the check): "
      f"{len(excluded_periods)} periods -> {excluded_periods}")

# Final sanity check + summary, read back from disk (not from memory) so this
# reflects exactly what actually made it into the file.
final = pd.read_csv(OUTPUT_FILE, low_memory=False)
print(f"\nFinal file shape: {final.shape}")
print(f"Periods present: {final['period'].nunique()}")
print(final.head())