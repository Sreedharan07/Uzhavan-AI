"""
🌾 UZHAVAN AI - Research Modules Verification Suite
Tests all newly added research-grade components for conference publication:
1. Diglossia Normalizer
2. Dialect Prompt Conditioning
3. Acoustic-Phonetic Prosody Extractor & Model
4. Multimodal Agri-Consultation Logic
5. Verified Grounded RAG & AHI Guardrail
6. Benchmark & LaTeX Table Generation
"""

import sys
import os
from pathlib import Path

# Reconfigure stdout/stderr for Windows UTF-8 console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def test_diglossia_normalizer():
    print("Testing 1: Tamil Diglossia Normalizer...")
    from ml.nlp.diglossia_normalizer import get_diglossia_normalizer
    normalizer = get_diglossia_normalizer()

    # Test Kongu dialect sentence with agricultural slang
    sample = "நெல்லுக்கு என்ன உரம் போடணுங்க? வயல்ல செம்பேன் பூச்சி இருக்கு."
    res = normalizer.normalize(sample, dialect_hint="kongu")
    
    assert res["normalized"] != "", "Normalized text should not be empty"
    assert "போட வேண்டும்" in res["normalized"] or "இட வேண்டும்" in res["normalized"], "Colloquial verb should be normalized"
    assert len(res["terms_mapped"]) > 0, "Agronomic slang 'செம்பேன்' should be mapped"
    print(f"   Input:      {sample}")
    print(f"   Normalized: {res['normalized']}")
    print(f"   Mapped:     {res['terms_mapped']}")
    print("   ✅ Diglossia Normalizer test passed!\n")


def test_dialect_prompt_manager():
    print("Testing 2: Dialect Prompt Manager...")
    from ml.asr.dialect_conditioned_asr import DialectPromptManager

    prompt_kongu = DialectPromptManager.get_prompt_for_dialect("kongu")
    prompt_madurai = DialectPromptManager.get_prompt_for_dialect("madurai")
    prompt_nellai = DialectPromptManager.get_prompt_for_dialect("nellai")

    assert "கொங்கு" in prompt_kongu, "Kongu prompt should contain Kongu marker"
    assert "மதுரை" in prompt_madurai, "Madurai prompt should contain Madurai marker"
    assert "திருநெல்வேலி" in prompt_nellai, "Nellai prompt should contain Nellai marker"
    print(f"   Kongu Prompt:   {prompt_kongu[:40]}...")
    print(f"   Nellai Prompt:  {prompt_nellai[:40]}...")
    print("   ✅ Dialect Prompt Manager test passed!\n")


def test_prosody_features_and_model():
    print("Testing 3: Acoustic-Phonetic Prosody Features & Neural Model...")
    from ml.dialect.prosody_features import extract_prosodic_features
    from ml.dialect.prosody_model import ProsodyEnhancedDialectClassifier

    # Create 1 second of synthetic speech-like harmonic signal
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)

    prosody_vec = extract_prosodic_features(audio, sr=sr, feature_dim=32)
    assert prosody_vec.shape == (32,), f"Prosody vector shape should be (32,), got {prosody_vec.shape}"
    assert not np.isnan(prosody_vec).any(), "Prosody features should not contain NaNs"

    # Test PyTorch model forward pass
    model = ProsodyEnhancedDialectClassifier(n_mels=80, num_classes=10)
    mel_tensor = torch.randn(2, 1, 80, 150)  # batch size 2, 1 channel, 80 mels, 150 frames
    prosody_tensor = torch.randn(2, 32)

    with torch.no_grad():
        out = model(mel_tensor, prosody_tensor)
    assert out.shape == (2, 10), f"Model output shape should be (2, 10), got {out.shape}"
    print(f"   Extracted 32-d prosody vector: mean={np.mean(prosody_vec):.3f}, std={np.std(prosody_vec):.3f}")
    print(f"   Forward pass logits shape: {out.shape}")
    print("   ✅ Prosody Features and Model test passed!\n")


def test_multimodal_symptom_analyzer():
    print("Testing 4: Multimodal Agri-Consultant Symptom Analyzer...")
    from ml.crop_disease.multimodal_fusion import MultimodalAgriConsultant

    consultant = MultimodalAgriConsultant()
    text = "தக்காளி இலையில் மஞ்சள் புள்ளி மற்றும் கருகல் நோய் வந்திருக்கு"
    symptom_res = consultant.analyze_symptom_text(text)

    assert len(symptom_res["matched_symptoms"]) > 0, "Should detect symptom keywords"
    assert len(symptom_res["disease_scores"]) > 0, "Should compute disease scores"
    print(f"   Symptom query: {text}")
    print(f"   Identified:    {symptom_res['matched_symptoms']}")
    print(f"   Top candidate: {list(symptom_res['disease_scores'].keys())[:2]}")
    print("   ✅ Multimodal Symptom Analyzer test passed!\n")


def test_grounded_fact_checker():
    print("Testing 5: Grounded Fact-Checker & AHI Metric...")
    from rag.verified_rag import GroundedFactChecker

    checker = GroundedFactChecker()
    # Response with verified numbers vs unverified numbers
    sample_response = "நெற்பயிருக்கு ஏக்கருக்கு 120 கிலோ யூரியா இட வேண்டும். மற்றும் 500 மில்லி வேப்ப எண்ணெய் தெளிக்கவும்."
    docs = [{"text": "நெல் பயிர் உரம்: ஏக்கருக்கு 120 கிலோ யூரியா, 50 கிலோ டிஏபி."}]

    verification = checker.verify_response(sample_response, docs, topic="fertilizer")
    assert "ahi_score" in verification, "AHI score must be computed"
    assert "badge" in verification, "Verification badge must be generated"
    print(f"   Sample Response: {sample_response}")
    print(f"   AHI Score:       {verification['ahi_score']}")
    print(f"   Safety Badge:    {verification['badge']}")
    print(f"   Verified Claims: {verification['verified_claims']}")
    print("   ✅ Grounded Fact-Checker test passed!\n")


def test_paper_tables():
    print("Testing 6: Conference Table Generator...")
    from scripts.generate_paper_tables import generate_latex_tables
    t1, t2, t3, t4 = generate_latex_tables()
    assert r"\begin{table}" in t1, "Table 1 must be valid LaTeX"
    assert r"\begin{table}" in t3, "Table 3 must be valid LaTeX"
    print("   ✅ LaTeX publication tables generated successfully!\n")


def run_all_tests():
    print("=" * 60)
    print("🌾 UZHAVAN AI - Running Research Verification Test Suite")
    print("=" * 60)
    test_diglossia_normalizer()
    test_dialect_prompt_manager()
    test_prosody_features_and_model()
    test_multimodal_symptom_analyzer()
    test_grounded_fact_checker()
    test_paper_tables()
    print("=" * 60)
    print("🎉 ALL 6 RESEARCH MODULE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
