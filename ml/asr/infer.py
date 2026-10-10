"""
🌾 UZHAVAN AI - ASR Inference
Single-file and batch inference with the trained model.

Usage:
    python ml/asr/infer.py --audio path/to/audio.wav
    python ml/asr/infer.py --audio-dir path/to/audio_dir/
    python ml/asr/infer.py --audio path/to/audio.wav --model models/asr/best_model
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def transcribe(
    audio_path: str,
    model_dir: str = "models/asr/best_model",
    architecture: str = "whisper",
    device: Optional[str] = None,
    _cached_model: Optional[tuple] = None,
    dialect: Optional[str] = None,
    apply_diglossia_normalization: bool = True,
) -> Dict:
    """
    Transcribe a single audio file with optional dialect conditioning and diglossia normalization.
    
    Returns:
        {
            "transcription": str,
            "normalized_transcription": str,
            "language": str,
            "latency_ms": float,
            "audio_duration": float,
            "rtf": float,
        }
    """
    import librosa
    from ml.models.model_factory import load_trained_model
    from ml.nlp.diglossia_normalizer import get_diglossia_normalizer
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # If dialect is explicitly provided, we can use DialectConditionedASR
    if dialect:
        try:
            from ml.asr.dialect_conditioned_asr import get_dialect_conditioned_asr
            engine = get_dialect_conditioned_asr()
            res = engine.transcribe_with_dialect(audio_path, dialect_hint=dialect)
            return {
                "transcription": res["verbatim_transcription"],
                "normalized_transcription": res["normalized_transcription"],
                "language": "ta",
                "latency_ms": res["latency_ms"],
                "audio_duration": res["audio_duration"],
                "rtf": round(res["latency_ms"] / 1000 / max(0.01, res["audio_duration"]), 4),
                "dialect": res["dialect"],
            }
        except Exception:
            pass  # Fall back to standard pipeline

    # Load model (or use cached)
    if _cached_model:
        model, processor = _cached_model
    else:
        model, processor = load_trained_model(model_dir, architecture, device)
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Preprocess
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    
    # Inference with timing
    start_time = time.time()
    with torch.no_grad():
        predicted_ids = model.generate(input_features=input_features, max_new_tokens=225)
    latency = (time.time() - start_time) * 1000
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    normalized_transcription = transcription
    if apply_diglossia_normalization:
        try:
            normalizer = get_diglossia_normalizer()
            norm_res = normalizer.normalize(transcription, dialect_hint=dialect)
            normalized_transcription = norm_res["normalized"]
        except Exception:
            pass

    return {
        "transcription": transcription,
        "normalized_transcription": normalized_transcription,
        "language": "ta",
        "latency_ms": round(latency, 2),
        "audio_duration": round(len(audio) / sr, 2),
        "rtf": round(latency / 1000 / max(0.01, (len(audio) / sr)), 4),  # Real-time factor
    }


def transcribe_batch(
    audio_dir: str,
    model_dir: str = "models/asr/best_model",
    architecture: str = "whisper",
) -> list:
    """Transcribe all audio files in a directory."""
    from ml.models.model_factory import load_trained_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_trained_model(model_dir, architecture, device)
    cached = (model, processor)
    
    audio_files = list(Path(audio_dir).glob("*.wav")) + list(Path(audio_dir).glob("*.mp3"))
    results = []
    
    for audio_file in audio_files:
        result = transcribe(
            str(audio_file),
            model_dir=model_dir,
            architecture=architecture,
            device=device,
            _cached_model=cached,
        )
        result["file"] = audio_file.name
        results.append(result)
        print(f"  {audio_file.name}: {result['transcription']} ({result['latency_ms']:.0f}ms)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="🌾 UZHAVAN AI - ASR Inference")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--audio-dir", type=str, help="Path to directory of audio files")
    parser.add_argument("--model", type=str, default="models/asr/best_model", help="Model directory")
    parser.add_argument("--architecture", type=str, default="whisper", help="Model architecture")
    args = parser.parse_args()
    
    if args.audio:
        result = transcribe(args.audio, args.model, args.architecture)
        print(f"\n🎤 Transcription: {result['transcription']}")
        print(f"   Latency: {result['latency_ms']:.0f}ms")
        print(f"   Audio duration: {result['audio_duration']}s")
        print(f"   RTF: {result['rtf']}")
    elif args.audio_dir:
        results = transcribe_batch(args.audio_dir, args.model, args.architecture)
        print(f"\n📊 Transcribed {len(results)} files")
        avg_latency = np.mean([r["latency_ms"] for r in results])
        print(f"   Average latency: {avg_latency:.0f}ms")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
