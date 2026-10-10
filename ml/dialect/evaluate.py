"""
🌾 UZHAVAN AI - Dialect Classifier Evaluation & Inference

Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
Inference: Returns {"dialect": "Kongu Tamil", "confidence": 0.91}
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

DIALECT_DISPLAY_NAMES = {
    "standard": "Standard Tamil",
    "kongu": "Kongu Tamil",
    "madurai": "Madurai Tamil",
    "tirunelveli": "Tirunelveli Tamil",
    "nellai": "Nellai Tamil",
    "thanjavur": "Delta Tamil",
    "chennai": "Chennai Tamil",
    "northern": "Northern Tamil",
    "southern": "Southern Tamil",
    "code_mixed": "Tamil-English Code-Mixed",
}


def load_dialect_model(model_path: str, device: Optional[str] = None):
    """Load trained dialect model."""
    from ml.dialect.models import create_dialect_model
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(model_path, map_location=device)
    
    architecture = checkpoint.get("architecture", "cnn_bilstm")
    config = checkpoint.get("config", {})
    dialect_labels = checkpoint.get("dialect_labels", list(DIALECT_DISPLAY_NAMES.keys()))
    
    if architecture == "prosody_cnn_bilstm":
        from ml.dialect.prosody_model import ProsodyEnhancedDialectClassifier
        model = ProsodyEnhancedDialectClassifier(
            n_mels=config.get("n_mels", 80),
            num_classes=len(dialect_labels),
        )
    else:
        model = create_dialect_model(architecture, config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, dialect_labels


def classify_dialect(
    audio_path: str,
    model_path: str = "models/dialect/best_dialect_model.pt",
    device: Optional[str] = None,
    _cached: Optional[tuple] = None,
    include_prosody_features: bool = True,
) -> Dict:
    """
    Classify the dialect of a Tamil audio file.
    
    Returns:
        {"dialect": "Kongu Tamil", "confidence": 0.91, "all_scores": {...}, "prosody_stats": {...}}
    """
    import librosa
    from ml.dialect.prosody_features import extract_prosodic_features
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if _cached:
        model, dialect_labels = _cached
    else:
        model, dialect_labels = load_dialect_model(model_path, device)
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Fixed segment (5 seconds)
    segment_samples = 5 * 16000
    if len(audio) > segment_samples:
        audio_segment = audio[:segment_samples]
    else:
        audio_segment = np.pad(audio, (0, segment_samples - len(audio)))
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_segment, sr=16000, n_mels=80, n_fft=1024, hop_length=512)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    mel_tensor = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0).to(device)
    
    # Extract prosody features
    prosody_vec = extract_prosodic_features(audio, sr=16000, feature_dim=32)
    prosody_tensor = torch.FloatTensor(prosody_vec).unsqueeze(0).to(device)

    # Predict (model may accept prosody or mel alone)
    with torch.no_grad():
        try:
            logits = model(mel_tensor, prosody_tensor)
        except TypeError:
            logits = model(mel_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
    
    pred_idx = probs.argmax().item()
    pred_dialect = dialect_labels[pred_idx]
    confidence = probs[pred_idx].item()
    
    all_scores = {
        DIALECT_DISPLAY_NAMES.get(d, d): float(probs[i])
        for i, d in enumerate(dialect_labels)
    }
    
    res = {
        "dialect": DIALECT_DISPLAY_NAMES.get(pred_dialect, pred_dialect),
        "dialect_code": pred_dialect,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    }
    if include_prosody_features:
        res["prosody_features"] = {
            "pitch_norm": round(float(prosody_vec[0]), 3),
            "rms_norm": round(float(prosody_vec[7]), 3),
            "spectral_centroid_norm": round(float(prosody_vec[10]), 3),
            "jitter_proxy": round(float(prosody_vec[16]), 3),
            "shimmer_proxy": round(float(prosody_vec[17]), 3),
        }
    return res


def evaluate_dialect_model(
    model_path: str = "models/dialect/best_dialect_model.pt",
    test_manifest: str = "datasets/tamil_rural/test/metadata.csv",
    results_dir: str = "results/dialect",
):
    """Evaluate dialect classifier on test set."""
    import csv
    from sklearn.metrics import classification_report, confusion_matrix
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, dialect_labels = load_dialect_model(model_path, device)
    cached = (model, dialect_labels)
    
    manifest = Path(test_manifest)
    audio_dir = manifest.parent / "audio"
    
    y_true, y_pred = [], []
    
    with open(manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = audio_dir / row["audio_path"]
            if not audio_path.exists():
                continue
            
            true_dialect = row.get("dialect", "standard")
            if true_dialect not in dialect_labels:
                continue
            
            result = classify_dialect(str(audio_path), device=device, _cached=cached)
            pred_dialect = result["dialect_code"]
            
            y_true.append(true_dialect)
            y_pred.append(pred_dialect)
    
    if not y_true:
        print("❌ No test samples found!")
        return
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        labels=dialect_labels,
        target_names=[DIALECT_DISPLAY_NAMES.get(d, d) for d in dialect_labels],
        zero_division=0,
        output_dict=True,
    )
    
    # Save results
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "dialect_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print
    print("\n" + "=" * 60)
    print("📊 Dialect Classification Results")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        labels=dialect_labels,
        target_names=[DIALECT_DISPLAY_NAMES.get(d, d) for d in dialect_labels],
        zero_division=0,
    ))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/dialect/best_dialect_model.pt")
    parser.add_argument("--test", default="datasets/tamil_rural/test/metadata.csv")
    parser.add_argument("--audio", default=None, help="Single audio file to classify")
    args = parser.parse_args()
    
    if args.audio:
        result = classify_dialect(args.audio, args.model)
        print(f"\n🗣️ Dialect: {result['dialect']}")
        print(f"   Confidence: {result['confidence']:.2%}")
    else:
        evaluate_dialect_model(args.model, args.test)
