"""Test script for 50x Fast XGBoost Crop Recommender."""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from ml.crop_calendar.xgboost_recommender import get_crop_recommender

print("🌾 Initializing XGBoost Crop Recommender...")
rec = get_crop_recommender()

# 1. Recommendation Test
print("\n--- Test 1: Recommendation (Thanjavur Delta, Samba Season) ---")
res = rec.recommend(district="thanjavur", month=8, water_source="canal")
print(f"Region: {res['region']}, Season: {res['season']} ({res['season_ta']})")
print(f"Inference Latency: {res['performance']['inference_time_us']} µs ({res['performance']['engine']})")
print("Top 3 Crops Recommended:")
for c in res['top_crops'][:3]:
    print(f"  {c['rank']}. {c['crop']} ({c['crop_en']}) - Confidence: {c['confidence_pct']} | Yield: {c['yield_per_acre']} | Duration: {c['duration_days']} days")

# 2. Recommendation Test with Custom Soil Sensor Data
print("\n--- Test 2: Custom Soil Parameters (Kongu Region) ---")
res2 = rec.recommend(district="kongu", n=70, p=45, k=60, ph=6.8, month=6, soil_type="red", water_source="borewell")
print("Top 3 Crops:")
for c in res2['top_crops'][:3]:
    print(f"  {c['rank']}. {c['crop']} ({c['crop_en']}) - Confidence: {c['confidence_pct']}")

# 3. Live Benchmark (Standard XGBoost vs 50x Fast JIT)
print("\n--- Test 3: Live Acceleration Benchmark (300 iterations) ---")
bm = rec.benchmark_speedup(iterations=300)
std_mean = bm["standard_xgboost"]["mean_us"]
fast_mean = bm["fast_jit_xgboost"]["mean_us"]
speedup = bm["speedup_factor"]

print(f"Standard XGBoost Latency: {std_mean:.2f} µs ({std_mean/1000:.4f} ms)")
print(f"Fast JIT Flattened Engine: {fast_mean:.2f} µs ({fast_mean/1000:.4f} ms)")
print(f"Speedup Factor: {speedup:.1f}x FASTER!")
print(f"Throughput: {bm['fast_jit_xgboost']['throughput_qps']:,} inferences/second")
print(f"Numerical Equivalence Verified: {bm['numerical_equivalence']} (Max diff: {bm['max_prediction_diff']:.6f})")

assert speedup >= 20.0, f"Speedup {speedup}x is below expected threshold!"
assert len(res['top_crops']) >= 3, "Failed to get top crops"
print("\n✅ All 50x Fast XGBoost tests PASSED successfully!")
