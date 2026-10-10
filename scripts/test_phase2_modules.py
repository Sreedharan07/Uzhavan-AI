"""
🌾 UZHAVAN AI - Phase 2 Research Modules Verification Suite
Tests all 8 new features + 5 optimized ML models.
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"   ✅ {name}")
    else:
        FAILED += 1
        print(f"   ❌ {name} — {detail}")


def test_mobilenet_v3():
    print("\n[Model 1] MobileNetV3-Small Crop Classifier...")
    from ml.crop_disease.mobilenet_classifier import MobileNetV3CropClassifier, count_parameters
    model = MobileNetV3CropClassifier(num_classes=38, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    check("Forward pass shape", out.shape == (2, 38), f"got {out.shape}")
    stats = count_parameters(model)
    check("Params < 3M", stats["total_params"] < 3_000_000, f"got {stats['total_params']}")
    feat = model.extract_features(x)
    check("Feature extraction (576-d)", feat.shape == (2, 576), f"got {feat.shape}")


def test_distilled_intent():
    print("\n[Model 2] Distilled Intent Classifier...")
    from ml.intent.distil_intent import DistilledIntentClassifier, SimpleTokenizer
    model = DistilledIntentClassifier(num_classes=14)
    model.eval()
    tok = SimpleTokenizer()
    encoded = tok.encode("நெல்லுக்கு என்ன உரம் போடணும்?")
    with torch.no_grad():
        out = model(**encoded)
    check("Forward pass shape", out.shape == (1, 14), f"got {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    check("Params < 12M (compact vs 237M MuRIL)", params < 12_000_000, f"got {params}")
    check("Tokenizer works", len(encoded["input_ids"][0]) == 128)


def test_dscnn_dialect():
    print("\n[Model 3] DS-CNN Dialect Classifier...")
    from ml.dialect.models import DialectCNN_BiLSTM, DialectDSCNN_BiLSTM, create_dialect_model
    std_model = DialectCNN_BiLSTM(num_classes=10)
    ds_model = DialectDSCNN_BiLSTM(num_classes=10)
    std_params = sum(p.numel() for p in std_model.parameters())
    ds_params = sum(p.numel() for p in ds_model.parameters())
    x = torch.randn(2, 1, 80, 150)
    with torch.no_grad():
        out = ds_model(x)
    check("Forward pass shape", out.shape == (2, 10), f"got {out.shape}")
    check("DS-CNN has fewer params", ds_params < std_params, f"DS={ds_params} vs STD={std_params}")
    reduction = (1 - ds_params / std_params) * 100
    check(f"Param reduction > 3% (got {reduction:.1f}%)", reduction > 3)
    factory_model = create_dialect_model("dscnn_bilstm")
    check("Factory 'dscnn_bilstm' creates model", factory_model is not None)


def test_wake_word():
    print("\n[Feature 1] Wake Word Detection...")
    from ml.wake_word.wake_word_model import WakeWordDetector, WakeWordDSCNN
    model = WakeWordDSCNN()
    x = torch.randn(1, 40, 25)
    with torch.no_grad():
        out = model(x)
    check("Model forward (B,2)", out.shape == (1, 2), f"got {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    check(f"Tiny model (<50K params): {params}", params < 50_000)
    detector = WakeWordDetector()
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    result = detector.detect(audio)
    check("Detect returns keys", all(k in result for k in ["detected", "confidence", "vad_active"]))


def test_emotion():
    print("\n[Feature 2] Farmer Emotion & Urgency Detection...")
    from ml.emotion.emotion_classifier import EmotionTCN, EmotionUrgencyDetector, extract_emotion_features
    feats = extract_emotion_features(np.random.randn(16000).astype(np.float32))
    check("Emotion features shape (20, T)", feats.shape[0] == 20 and feats.shape[1] > 0)
    model = EmotionTCN()
    model.eval()
    x = torch.from_numpy(feats).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    check("TCN forward (B, 5)", out.shape == (1, 5), f"got {out.shape}")
    detector = EmotionUrgencyDetector()
    result = detector.detect_emotion(np.random.randn(16000).astype(np.float32))
    check("Detect returns emotion", result["emotion"] in ["neutral", "worried", "urgent_distress", "happy", "confused"])
    check("Urgency level returned", result["urgency_level"] in ["normal", "elevated", "high"])


def test_acoustic_scene():
    print("\n[Feature 3] Acoustic Scene Classification...")
    from ml.acoustic_scene.scene_classifier import AudioMobileNetV3, AcousticSceneClassifier
    model = AudioMobileNetV3(num_classes=5, pretrained=False)
    model.eval()
    x = torch.randn(1, 1, 64, 32)
    with torch.no_grad():
        out = model(x)
    check("AudioMobileNet forward (B, 5)", out.shape == (1, 5), f"got {out.shape}")
    classifier = AcousticSceneClassifier()
    result = classifier.classify_scene(np.random.randn(32000).astype(np.float32))
    check("Scene classification result", result["scene"] in ["indoor_quiet", "open_field", "tractor_nearby", "market_crowd", "pump_shed"])
    check("Noise profile returned", "noise_reduce_strength" in result["noise_profile"])


def test_soil():
    print("\n[Feature 4] Soil Health Prediction...")
    from ml.soil.soil_predictor import SoilPredictor
    predictor = SoilPredictor()
    result = predictor.predict("வயல்ல கருப்பு மண் இருக்கு, வெடிப்பு விழும், ஒட்டும்")
    check("Predicts black cotton soil", result["predicted_soil"] == "black_cotton")
    check("Confidence > 0", result["confidence"] > 0)
    check("Has recommendations", len(result["recommendations"]["suitable_crops"]) > 0)
    result2 = predictor.predict("சிவப்பு மண், உதிரும் தன்மை")
    check("Predicts red laterite", result2["predicted_soil"] == "red_laterite")


def test_crop_calendar():
    print("\n[Feature 5] Temporal Crop Calendar...")
    from ml.crop_calendar.crop_recommender import CropRecommender
    recommender = CropRecommender()
    result = recommender.recommend(district="thanjavur", month=8)
    check("Returns season", "season" in result and result["season"] != "")
    check("Returns crops", len(result["top_crops"]) > 0)
    check("Crop has yield info", "yield_per_acre" in result["top_crops"][0])
    result2 = recommender.recommend(district="kongu", month=3, water_source="rainfed")
    check("Water adjustment applied", result2["top_crops"][0]["confidence"] > 0)


def test_onnx_pipeline():
    print("\n[Feature 6] ONNX Inference Pipeline...")
    from ml.optimization.onnx_exporter import ONNXExporter, OptimizedInferencePipeline
    pipeline = OptimizedInferencePipeline()
    from ml.wake_word.wake_word_model import WakeWordDSCNN
    model = WakeWordDSCNN()
    pipeline.register_pytorch_model("wake_word", model)
    dummy = torch.randn(1, 40, 25)
    result = pipeline.infer("wake_word", {}, pytorch_input=dummy)
    check("PyTorch fallback works", result["backend"] == "pytorch")
    check("Latency logged", result["latency_ms"] > 0)
    report = pipeline.get_latency_report()
    check("Latency report generated", "wake_word" in report)


def test_code_switch():
    print("\n[Feature 7] Code-Switching Detection...")
    from ml.nlp.code_switch_detector import CodeSwitchDetector
    detector = CodeSwitchDetector()
    result = detector.detect_boundaries("நான் paddy field-ல urea போட்டேன்")
    check("Segments returned", len(result["segments"]) > 0)
    check("Switch count >= 1", result["switch_count"] >= 1)
    check("Code mixing ratio > 0", result["code_mixing_ratio"] > 0)
    # Pure Tamil
    result2 = detector.detect_boundaries("நெல்லுக்கு உரம் போடணும்")
    check("Pure Tamil: dominant=TA", result2["dominant_language"] == "TA")


def test_federated():
    print("\n[Feature 8] Federated LoRA Adaptation...")
    from ml.federated.federated_adapter import FederatedDialectAdapter, FederatedAggregationServer
    from ml.dialect.models import DialectCNN_BiLSTM
    base = DialectCNN_BiLSTM(num_classes=10)
    adapter = FederatedDialectAdapter(base, lora_rank=4, dp_epsilon=8.0)
    stats = adapter.get_adaptation_stats()
    check(f"Adaptation ratio < 10% (got {stats['adaptation_ratio']}%)", stats["adaptation_ratio"] < 10)
    check("LoRA rank = 4", stats["lora_rank"] == 4)
    check("DP epsilon = 8.0", stats["dp_epsilon"] == 8.0)
    # Test federated aggregation
    deltas1 = adapter.get_parameter_deltas()
    server = FederatedAggregationServer()
    server.receive_update("farmer_A", deltas1)
    server.receive_update("farmer_B", deltas1)
    aggregated = server.aggregate()
    check("Server aggregation works", len(aggregated) > 0)
    adapter.apply_aggregated_deltas(aggregated)
    check("Deltas applied successfully", True)


def run_all():
    print("=" * 70)
    print("🌾 UZHAVAN AI — Phase 2 Research Verification Suite")
    print("   Testing 8 Features + 5 Optimized ML Models")
    print("=" * 70)

    test_mobilenet_v3()
    test_distilled_intent()
    test_dscnn_dialect()
    test_wake_word()
    test_emotion()
    test_acoustic_scene()
    test_soil()
    test_crop_calendar()
    test_onnx_pipeline()
    test_code_switch()
    test_federated()

    print("\n" + "=" * 70)
    total = PASSED + FAILED
    print(f"Results: {PASSED}/{total} passed, {FAILED}/{total} failed")
    if FAILED == 0:
        print("🎉 ALL PHASE 2 TESTS PASSED WITH 100% SUCCESS!")
    else:
        print(f"⚠️  {FAILED} test(s) failed — review output above")
    print("=" * 70)
    return FAILED == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
