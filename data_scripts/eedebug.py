"""
Quick Earth Engine health check
=================================
Run this on its own -- small, fast, no dependency on your dataset files.
Tells you whether Earth Engine is actually quota-throttled right now,
or working fine (meaning your earlier run was interrupted by something else).
"""

import ee

PROJECT_ID = ""

print(f"Initializing Earth Engine with project '{PROJECT_ID}'...")
try:
    ee.Initialize(project=PROJECT_ID)
    print("Initialized OK.\n")
except Exception as e:
    print(f"FAILED to initialize at all: {e}")
    print("This suggests an auth problem, not a quota problem -- try ee.Authenticate() again.")
    raise SystemExit

# Test 1: trivial computation, no data access at all
print("Test 1: trivial computation (no real data access)...")
try:
    result = ee.Number(1).add(1).getInfo()
    print(f"  OK -- got {result} back.\n")
except Exception as e:
    print(f"  FAILED: {e}")
    print("  If this fails, something more basic than quota is wrong (auth/project setup).\n")

# Test 2: a small real reduceRegions call, similar to what the main script does
print("Test 2: small real Earth Engine data call (one GLDAS image, one point)...")
try:
    point = ee.Geometry.Point([78.1639, 14.1917]).buffer(5000)
    img = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
            .filterDate("2020-01-01", "2020-01-31") \
            .select("SoilMoi0_10cm_inst").mean()
    fc = ee.FeatureCollection([ee.Feature(point, {"well_id": "test"})])
    out = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=27830).getInfo()
    print(f"  OK -- got result: {out['features'][0]['properties']}\n")
except Exception as e:
    msg = str(e)
    print(f"  FAILED: {msg}\n")
    if "quota" in msg.lower() or "restricted" in msg.lower() or "rate" in msg.lower():
        print("  ^ This looks like an actual QUOTA/RATE LIMIT error.")
        print("  You likely need to wait for the monthly reset, or check your usage at:")
        print("  https://code.earthengine.google.com/ -> your project -> Quotas")
    else:
        print("  ^ This does NOT look like a quota error -- likely something else")
        print("  (auth expired, network issue, or a genuine bug). Read the message above.")

print("Done.")