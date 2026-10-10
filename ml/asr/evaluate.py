"""
🌾 UZHAVAN AI - ASR Evaluation
Comprehensive evaluation with dialect-wise, domain-wise metrics.

Metrics:
    - WER (Word Error Rate)
    - CER (Character Error Rate)
    - Dialect-wise WER breakdown
    - Agriculture vocabulary accuracy
    - Code-mixed WER
    - Per-speaker-group WER (elderly, adult)

Usage:
    python ml/asr/evaluate.py --model models/asr/best_model --test datasets/tamil_rural/test/metadata.csv
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_asr(
    model_dir: str,
    test_manifest: str,
    architecture: str = "whisper",
    results_dir: str = "results/asr",
    agriculture_vocab_path: str = "agriculture/tamil_agri_dictionary.json",
) -> Dict:
    """
    Run comprehensive ASR evaluation.
    
    Returns dict with all metrics.
    """
    from ml.models.model_factory import load_trained_model, print_device_info
    from jiwer import wer, cer
    import librosa
    
    device_info = print_device_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"\n📦 Loading model from: {model_dir}")
    model, processor = load_trained_model(model_dir, architecture, str(device))
    
    # Load agriculture vocabulary for domain-specific evaluation
    agri_vocab = {}
    agri_vocab_path = PROJECT_ROOT / agriculture_vocab_path
    if agri_vocab_path.exists():
        with open(agri_vocab_path, "r", encoding="utf-8") as f:
            agri_vocab = json.load(f)
    
    # Load test data
    manifest = Path(test_manifest)
    audio_dir = manifest.parent / "audio"
    
    samples = []
    with open(manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    
    print(f"📊 Evaluating on {len(samples)} test samples...")
    
    # Collect predictions grouped by metadata
    all_results = []
    by_dialect = defaultdict(lambda: {"preds": [], "refs": []})
    by_domain = defaultdict(lambda: {"preds": [], "refs": []})
    by_age = defaultdict(lambda: {"preds": [], "refs": []})
    by_code_mixed = defaultdict(lambda: {"preds": [], "refs": []})
    agri_preds = []
    agri_refs = []
    
    for sample in tqdm(samples, desc="Evaluating"):
        audio_path = audio_dir / sample["audio_path"]
        if not audio_path.exists():
            continue
        
        reference = sample.get("transcript", "").strip()
        if not reference:
            continue
        
        # Load and transcribe
        try:
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            input_features = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            
            with torch.no_grad():
                predicted_ids = model.generate(input_features=input_features, max_new_tokens=225)
            
            prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        except Exception as e:
            prediction = ""
        
        # Store results
        dialect = sample.get("dialect", "standard")
        domain = sample.get("domain", "general")
        age_group = sample.get("age_group", "adult")
        code_mixed = sample.get("code_mixed", "false")
        
        result = {
            "audio": sample["audio_path"],
            "reference": reference,
            "prediction": prediction,
            "dialect": dialect,
            "domain": domain,
            "age_group": age_group,
        }
        all_results.append(result)
        
        # Group by metadata
        by_dialect[dialect]["preds"].append(prediction)
        by_dialect[dialect]["refs"].append(reference)
        
        by_domain[domain]["preds"].append(prediction)
        by_domain[domain]["refs"].append(reference)
        
        by_age[age_group]["preds"].append(prediction)
        by_age[age_group]["refs"].append(reference)
        
        cm_key = "code_mixed" if code_mixed.lower() == "true" else "monolingual"
        by_code_mixed[cm_key]["preds"].append(prediction)
        by_code_mixed[cm_key]["refs"].append(reference)
        
        # Check agriculture vocabulary
        for tamil_term in agri_vocab.keys():
            if tamil_term in reference:
                agri_preds.append(prediction)
                agri_refs.append(reference)
                break
    
    # Compute overall metrics
    all_preds = [r["prediction"] for r in all_results]
    all_refs = [r["reference"] for r in all_results]
    
    overall_wer = wer(all_refs, all_preds) if all_refs else -1
    overall_cer = cer(all_refs, all_preds) if all_refs else -1
    
    # Compute dialect-wise metrics
    dialect_metrics = {}
    for dialect, data in by_dialect.items():
        if data["refs"]:
            dialect_metrics[dialect] = {
                "wer": wer(data["refs"], data["preds"]),
                "cer": cer(data["refs"], data["preds"]),
                "num_samples": len(data["refs"]),
            }
    
    # Compute domain-wise metrics
    domain_metrics = {}
    for domain, data in by_domain.items():
        if data["refs"]:
            domain_metrics[domain] = {
                "wer": wer(data["refs"], data["preds"]),
                "num_samples": len(data["refs"]),
            }
    
    # Compute age-group metrics
    age_metrics = {}
    for age, data in by_age.items():
        if data["refs"]:
            age_metrics[age] = {
                "wer": wer(data["refs"], data["preds"]),
                "num_samples": len(data["refs"]),
            }
    
    # Code-mixed metrics
    code_mixed_metrics = {}
    for cm_type, data in by_code_mixed.items():
        if data["refs"]:
            code_mixed_metrics[cm_type] = {
                "wer": wer(data["refs"], data["preds"]),
                "num_samples": len(data["refs"]),
            }
    
    # Agriculture vocabulary accuracy
    agri_wer = wer(agri_refs, agri_preds) if agri_refs else -1
    
    # Compile results
    results = {
        "overall": {
            "wer": overall_wer,
            "cer": overall_cer,
            "total_samples": len(all_results),
        },
        "dialect_wise": dialect_metrics,
        "domain_wise": domain_metrics,
        "age_group_wise": age_metrics,
        "code_mixed": code_mixed_metrics,
        "agriculture_vocab_wer": agri_wer,
        "model_dir": model_dir,
        "test_manifest": test_manifest,
    }
    
    # Save results
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save predictions
    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Save dialect metrics CSV
    with open(out_dir / "dialect_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dialect", "WER", "CER", "Num Samples"])
        for dialect, m in sorted(dialect_metrics.items()):
            writer.writerow([
                dialect,
                f"{m['wer']:.4f}",
                f"{m['cer']:.4f}",
                m["num_samples"],
            ])
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 ASR Evaluation Results")
    print("=" * 60)
    print(f"\n  Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"  Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    print(f"  Total samples: {len(all_results)}")
    
    if dialect_metrics:
        print(f"\n  {'Dialect':<20} {'WER':>8} {'CER':>8} {'Samples':>8}")
        print(f"  {'-'*46}")
        for dialect, m in sorted(dialect_metrics.items()):
            print(f"  {dialect:<20} {m['wer']*100:>7.2f}% {m['cer']*100:>7.2f}% {m['num_samples']:>8}")
    
    if agri_wer >= 0:
        print(f"\n  Agriculture Vocab WER: {agri_wer*100:.2f}%")
    
    print(f"\n  Results saved to: {out_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="🌾 UZHAVAN AI - Evaluate ASR Model")
    parser.add_argument("--model", type=str, default="models/asr/best_model", help="Model directory")
    parser.add_argument("--test", type=str, default="datasets/tamil_rural/test/metadata.csv", help="Test manifest")
    parser.add_argument("--architecture", type=str, default="whisper", help="Model architecture")
    parser.add_argument("--results-dir", type=str, default="results/asr", help="Results output directory")
    args = parser.parse_args()
    
    evaluate_asr(
        model_dir=args.model,
        test_manifest=args.test,
        architecture=args.architecture,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
