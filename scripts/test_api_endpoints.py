"""
🌾 UZHAVAN AI - API Endpoints Integration Test
Tests FastAPI backend application with new research endpoints using TestClient.
"""

import sys
import os
from pathlib import Path

# Reconfigure stdout/stderr for Windows UTF-8 console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_endpoints():
    print("Testing API routes...")

    # 1. Root and health
    res = client.get("/")
    assert res.status_code == 200, f"Root failed: {res.status_code}"
    print("   ✅ GET / -> 200 OK")

    # 2. Research benchmark summary
    res = client.get("/api/research/benchmark-summary")
    assert res.status_code == 200, f"Benchmark summary failed: {res.status_code}"
    data = res.json()
    assert "asr_wer_results" in data, "asr_wer_results missing"
    assert "multimodal_crop_diagnosis_accuracy" in data, "multimodal accuracy missing"
    print("   ✅ GET /api/research/benchmark-summary -> 200 OK")

    # 3. Assistant query with diglossia & verification
    res = client.post("/api/assistant/query?text=நெல்லுக்கு என்ன உரம் போடணுங்க?")
    assert res.status_code == 200, f"Assistant query failed: {res.status_code}"
    resp_data = res.json()
    assert "response" in resp_data, "response field missing"
    assert "verification_badge" in resp_data, "verification_badge field missing"
    assert "normalized_transcription" in resp_data, "normalized_transcription missing"
    print(f"   Query: 'நெல்லுக்கு என்ன உரம் போடணுங்க?'")
    print(f"   Normalized: '{resp_data.get('normalized_transcription')}'")
    print(f"   Badge: {resp_data.get('verification_badge')}")
    print("   ✅ POST /api/assistant/query -> 200 OK")

    print("\n🎉 ALL API ENDPOINTS VERIFIED AND FUNCTIONING PROPERLY!")


if __name__ == "__main__":
    test_endpoints()
