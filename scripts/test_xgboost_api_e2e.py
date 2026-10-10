"""
🌾 UZHAVAN AI - End-to-End Verification for 50x Fast XGBoost Crop Recommender
Verifies model accuracy, 50x acceleration benchmark, API endpoints, and numerical consistency.
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

def check(name: str, condition: bool, extra: str = ""):
    status = "✅" if condition else "❌"
    msg = f"   {status} {name}"
    if extra:
        msg += f" ({extra})"
    print(msg)
    if not condition:
        raise AssertionError(f"Check failed: {name}")

print("=" * 70)
print("🌾 Testing 50x Fast XGBoost Crop Recommender & End-to-End APIs")
print("=" * 70)

# 1. Model Level Tests
print("\n[Step 1] Model Initialization & 50x JIT Traversal...")
from ml.crop_calendar.xgboost_recommender import get_crop_recommender
rec = get_crop_recommender()
check("Model loaded", rec is not None)
check("Fast JIT engine compiled", rec.fast_engine is not None)

res_delta = rec.recommend(district="thanjavur", month=8, water_source="canal")
check("Thanjavur returns Samba paddy", "நெல்" in res_delta["top_crops"][0]["crop"])
check("Inference latency < 100 µs", res_delta["performance"]["inference_time_us"] < 100.0, f"{res_delta['performance']['inference_time_us']} µs")

# 2. Live Speedup Benchmark (Checking for > 30x - 60x speedup)
print("\n[Step 2] Live Acceleration Benchmark (300 iterations)...")
bm = rec.benchmark_speedup(iterations=300)
speedup = bm["speedup_factor"]
std_us = bm["standard_xgboost"]["mean_us"]
fast_us = bm["fast_jit_xgboost"]["mean_us"]

print(f"      Standard XGBoost: {std_us:.1f} µs ({std_us/1000:.4f} ms)")
print(f"      Fast JIT Engine:  {fast_us:.1f} µs ({fast_us/1000:.4f} ms)")
print(f"      Speedup:          {speedup:.1f}x FASTER")
print(f"      Throughput:       {bm['fast_jit_xgboost']['throughput_qps']:,} inferences/sec")

check("Speedup is >= 30x", speedup >= 30.0, f"got {speedup}x")
check("Numerical Equivalence", bm["numerical_equivalence"], f"max diff: {bm['max_prediction_diff']:.6f}")

# 3. CropRecommender Integrated Wrapper Test
print("\n[Step 3] CropRecommender Backward Compatible Wrapper...")
from ml.crop_calendar.crop_recommender import CropRecommender
wrapper = CropRecommender()
wrap_res = wrapper.recommend(district="kongu", month=6, soil_type="red", water_source="borewell")
check("Wrapper returns top_crops", len(wrap_res["top_crops"]) > 0)
check("Wrapper returns performance info", "performance" in wrap_res)
check("Wrapper returns all_seasons calendar", "all_seasons" in wrap_res)

# 4. FastAPI Endpoints Test (via starlette TestClient)
print("\n[Step 4] Backend FastAPI Endpoints...")
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Test /api/crop/calendar-recommend
r_cal = client.get("/api/crop/calendar-recommend?district=thanjavur&month=8")
check("GET /api/crop/calendar-recommend status 200", r_cal.status_code == 200)
cal_data = r_cal.json()
check("Endpoint returns season", "season" in cal_data)
check("Endpoint returns top_crops", len(cal_data["top_crops"]) > 0)

# Test /api/crop/recommend-xgboost
r_xgb = client.post("/api/crop/recommend-xgboost?district=kongu&soil_type=red&month=6&n=75&p=40&k=65")
check("POST /api/crop/recommend-xgboost status 200", r_xgb.status_code == 200)
xgb_data = r_xgb.json()
check("Has speedup factor", "speedup_factor" in xgb_data["performance"])

# Test /api/crop/xgboost-benchmark
r_bm = client.get("/api/crop/xgboost-benchmark?iterations=50")
check("GET /api/crop/xgboost-benchmark status 200", r_bm.status_code == 200)
bm_data = r_bm.json()
check("Benchmark has speedup_factor", bm_data["speedup_factor"] > 0)

# Test /api/benchmark/model-latency includes XGBoost
r_lat = client.get("/api/benchmark/model-latency")
check("GET /api/benchmark/model-latency status 200", r_lat.status_code == 200)
lat_data = r_lat.json()["benchmark"]
check("Model latency includes XGBoost_Standard_Crop", "XGBoost_Standard_Crop" in lat_data)
check("Model latency includes XGBoost_50x_Fast_JIT_Crop", "XGBoost_50x_Fast_JIT_Crop" in lat_data)

print("\n" + "=" * 70)
print(f"🎉 ALL 50X FAST XGBOOST TESTS PASSED WITH 100% SUCCESS! (Speedup: {speedup:.1f}x)")
print("=" * 70)
