"""
🌾 UZHAVAN AI - Dataset Preparation Pipeline
Converts downloaded raw datasets into unified training format.

Output format:
    datasets/tamil_rural/{train,validation,test}/
        audio/           - 16kHz mono WAV files
        metadata.csv     - Unified metadata

Usage:
    python scripts/prepare_dataset.py --source all
    python scripts/prepare_dataset.py --source openslr_65
    python scripts/prepare_dataset.py --split-only
"""

import os
import sys
import csv
import argparse
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Metadata CSV columns
METADATA_COLUMNS = [
    "audio_path",
    "transcript",
    "dialect",
    "speaker_id",
    "age_group",
    "gender",
    "domain",
    "code_mixed",
    "duration",
    "source",
]


def ensure_audio_format(input_path: Path, output_path: Path, target_sr: int = 16000) -> Optional[float]:
    """Convert audio to 16kHz mono WAV. Returns duration in seconds."""
    try:
        import soundfile as sf
        import librosa
        
        # Load and resample
        audio, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
        
        # Skip very short or silent audio
        duration = len(audio) / target_sr
        if duration < 0.5:
            return None
        if np.max(np.abs(audio)) < 0.001:
            return None
        
        # Save as 16-bit PCM WAV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, target_sr, subtype="PCM_16")
        
        return duration
    except Exception as e:
        print(f"  ⚠️ Failed to process {input_path.name}: {e}")
        return None


def generate_speaker_id(source: str, original_id: str = "") -> str:
    """Generate anonymized speaker ID."""
    raw = f"{source}_{original_id}" if original_id else f"{source}_{random.randint(0, 99999)}"
    return f"spk_{hashlib.md5(raw.encode()).hexdigest()[:8]}"


def process_openslr_65(raw_dir: Path, output_entries: List[Dict]) -> int:
    """Process OpenSLR SLR65 (Crowdsourced Tamil Multi-speaker)."""
    dataset_dir = raw_dir / "openslr_65"
    if not dataset_dir.exists():
        print(f"  ⏩ OpenSLR SLR65 not found at {dataset_dir}, skipping.")
        return 0
    
    count = 0
    for gender_dir in ["ta_in_female", "ta_in_male"]:
        gender = "female" if "female" in gender_dir else "male"
        wav_dir = dataset_dir / gender_dir
        
        if not wav_dir.exists():
            continue
        
        # Look for transcription file
        tsv_file = wav_dir / "line_index.tsv"
        txt_files = list(wav_dir.glob("*.txt"))
        
        transcriptions = {}
        if tsv_file.exists():
            with open(tsv_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        transcriptions[parts[0]] = parts[1]
        
        for txt_file in txt_files:
            if txt_file.name == "line_index.tsv":
                continue
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        transcriptions[parts[0]] = parts[1]
        
        # Process audio files
        for wav_file in wav_dir.glob("**/*.wav"):
            file_id = wav_file.stem
            transcript = transcriptions.get(file_id, "")
            
            if not transcript:
                continue
            
            output_entries.append({
                "source_path": str(wav_file),
                "transcript": transcript,
                "dialect": "standard",  # No dialect labels in SLR65
                "speaker_id": generate_speaker_id("slr65", file_id.split("_")[0] if "_" in file_id else ""),
                "age_group": "adult",
                "gender": gender,
                "domain": "general",
                "code_mixed": "false",
                "source": "openslr_65",
            })
            count += 1
    
    print(f"  ✅ OpenSLR SLR65: {count} samples")
    return count


def process_openslr_127(raw_dir: Path, output_entries: List[Dict]) -> int:
    """Process OpenSLR SLR127 (IISc-MILE Tamil ASR Corpus)."""
    dataset_dir = raw_dir / "openslr_127"
    if not dataset_dir.exists():
        print(f"  ⏩ OpenSLR SLR127 not found at {dataset_dir}, skipping.")
        return 0
    
    count = 0
    for split_name in ["tam_train", "tam_test"]:
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            # Try without prefix
            split_dir = dataset_dir
        
        # Look for transcription files
        for trans_file in split_dir.glob("**/*.trans.txt"):
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) < 2:
                        continue
                    
                    file_id, transcript = parts
                    # Find corresponding audio
                    wav_path = trans_file.parent / f"{file_id}.wav"
                    flac_path = trans_file.parent / f"{file_id}.flac"
                    
                    audio_path = wav_path if wav_path.exists() else (flac_path if flac_path.exists() else None)
                    if not audio_path:
                        continue
                    
                    speaker_id = file_id.split("-")[0] if "-" in file_id else file_id[:6]
                    
                    output_entries.append({
                        "source_path": str(audio_path),
                        "transcript": transcript,
                        "dialect": "standard",
                        "speaker_id": generate_speaker_id("slr127", speaker_id),
                        "age_group": "adult",
                        "gender": "unknown",
                        "domain": "general",
                        "code_mixed": "false",
                        "source": "openslr_127",
                    })
                    count += 1
    
        # Also handle if metadata is in CSV format
        for csv_file in split_dir.glob("**/*.csv"):
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        audio_col = next((c for c in row.keys() if "path" in c.lower() or "audio" in c.lower() or "file" in c.lower()), None)
                        text_col = next((c for c in row.keys() if "text" in c.lower() or "transcript" in c.lower() or "sentence" in c.lower()), None)
                        
                        if audio_col and text_col and row.get(text_col):
                            audio_path = split_dir / row[audio_col]
                            if not audio_path.exists():
                                audio_path = split_dir / "audio" / row[audio_col]
                            
                            if audio_path.exists():
                                output_entries.append({
                                    "source_path": str(audio_path),
                                    "transcript": row[text_col],
                                    "dialect": "standard",
                                    "speaker_id": generate_speaker_id("slr127", row.get("speaker_id", "")),
                                    "age_group": row.get("age", "adult"),
                                    "gender": row.get("gender", "unknown"),
                                    "domain": "general",
                                    "code_mixed": "false",
                                    "source": "openslr_127",
                                })
                                count += 1
            except Exception:
                continue
    
    print(f"  ✅ OpenSLR SLR127: {count} samples")
    return count


def process_common_voice_hf(raw_dir: Path, output_entries: List[Dict]) -> int:
    """Process Common Voice Tamil from HuggingFace cache."""
    dataset_dir = raw_dir / "common_voice"
    if not dataset_dir.exists():
        print(f"  ⏩ Common Voice not found at {dataset_dir}, skipping.")
        return 0
    
    count = 0
    try:
        from datasets import load_from_disk
        
        # Try to load from HF cache
        for cache_dir in dataset_dir.glob("**/dataset_dict.json"):
            ds = load_from_disk(str(cache_dir.parent))
            for split in ds:
                for item in ds[split]:
                    audio_path = item.get("path", "")
                    transcript = item.get("sentence", "")
                    if audio_path and transcript:
                        output_entries.append({
                            "source_path": audio_path,
                            "transcript": transcript,
                            "dialect": "standard",
                            "speaker_id": generate_speaker_id("cv", item.get("client_id", "")),
                            "age_group": item.get("age", "adult"),
                            "gender": item.get("gender", "unknown"),
                            "domain": "general",
                            "code_mixed": "false",
                            "source": "common_voice",
                        })
                        count += 1
    except Exception as e:
        print(f"  ⚠️ Error processing Common Voice: {e}")
    
    print(f"  ✅ Common Voice: {count} samples")
    return count


def process_custom_recordings(raw_dir: Path, output_entries: List[Dict]) -> int:
    """Process custom recordings from the data collection tool."""
    custom_dir = PROJECT_ROOT / "data_collection" / "storage" / "recordings"
    metadata_file = PROJECT_ROOT / "data_collection" / "storage" / "metadata.csv"
    
    if not metadata_file.exists():
        print(f"  ⏩ No custom recordings found, skipping.")
        return 0
    
    count = 0
    with open(metadata_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = custom_dir / row.get("filename", "")
            if audio_path.exists():
                output_entries.append({
                    "source_path": str(audio_path),
                    "transcript": row.get("transcript", ""),
                    "dialect": row.get("dialect", "standard"),
                    "speaker_id": row.get("speaker_id", generate_speaker_id("custom")),
                    "age_group": row.get("age_group", "adult"),
                    "gender": row.get("gender", "unknown"),
                    "domain": row.get("domain", "general"),
                    "code_mixed": row.get("code_mixed", "false"),
                    "source": "custom",
                })
                count += 1
    
    print(f"  ✅ Custom recordings: {count} samples")
    return count


def split_dataset(
    entries: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/validation/test with stratification by dialect."""
    random.seed(seed)
    
    # Group by dialect for stratified split
    by_dialect = defaultdict(list)
    for entry in entries:
        by_dialect[entry["dialect"]].append(entry)
    
    train, val, test = [], [], []
    
    for dialect, dialect_entries in by_dialect.items():
        random.shuffle(dialect_entries)
        n = len(dialect_entries)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        train.extend(dialect_entries[:n_train])
        val.extend(dialect_entries[n_train:n_train + n_val])
        test.extend(dialect_entries[n_train + n_val:])
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def write_split(
    entries: List[Dict],
    split_dir: Path,
    target_sr: int = 16000,
) -> int:
    """Write a dataset split - convert audio and create metadata CSV."""
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = split_dir / "metadata.csv"
    written = 0
    
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        
        for i, entry in enumerate(tqdm(entries, desc=f"Processing {split_dir.name}")):
            source_path = Path(entry["source_path"])
            
            if not source_path.exists():
                continue
            
            # Output filename
            out_filename = f"audio_{written:06d}.wav"
            out_path = audio_dir / out_filename
            
            # Convert audio
            duration = ensure_audio_format(source_path, out_path, target_sr)
            if duration is None:
                continue
            
            writer.writerow({
                "audio_path": out_filename,
                "transcript": entry["transcript"],
                "dialect": entry["dialect"],
                "speaker_id": entry["speaker_id"],
                "age_group": entry["age_group"],
                "gender": entry["gender"],
                "domain": entry["domain"],
                "code_mixed": entry["code_mixed"],
                "duration": f"{duration:.2f}",
                "source": entry["source"],
            })
            written += 1
    
    return written


def main():
    parser = argparse.ArgumentParser(
        description="🌾 UZHAVAN AI - Prepare Tamil Speech Datasets"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "openslr_65", "openslr_127", "common_voice", "custom"],
        help="Which raw dataset source to process",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="datasets/raw",
        help="Directory containing raw downloaded datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/tamil_rural",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training split ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Target sample rate"
    )
    parser.add_argument(
        "--split-only", action="store_true", help="Only re-split existing processed data"
    )
    args = parser.parse_args()
    
    raw_dir = PROJECT_ROOT / args.raw_dir
    output_dir = PROJECT_ROOT / args.output_dir
    
    print("\n🌾 UZHAVAN AI - Dataset Preparation Pipeline")
    print("=" * 50)
    
    # Collect entries from all sources
    all_entries = []
    
    if not args.split_only:
        processors = {
            "openslr_65": process_openslr_65,
            "openslr_127": process_openslr_127,
            "common_voice": process_common_voice_hf,
            "custom": process_custom_recordings,
        }
        
        sources = list(processors.keys()) if args.source == "all" else [args.source]
        
        print(f"\n📂 Processing sources: {', '.join(sources)}")
        print(f"   Raw data directory: {raw_dir}")
        
        for source in sources:
            if source in processors:
                processors[source](raw_dir, all_entries)
    
    if not all_entries:
        print("\n⚠️  No data found to process!")
        print("   Run 'python scripts/download_datasets.py --dataset catalog' to see available datasets.")
        print("   Or run 'python data_collection/app.py' to collect custom recordings.")
        
        # Create empty structure
        for split in ["train", "validation", "test"]:
            split_dir = output_dir / split
            audio_dir = split_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_path = split_dir / "metadata.csv"
            if not metadata_path.exists():
                with open(metadata_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
                    writer.writeheader()
        
        print(f"\n📁 Created empty dataset structure at: {output_dir}")
        return
    
    print(f"\n📊 Total entries collected: {len(all_entries)}")
    
    # Print dialect distribution
    dialect_counts = defaultdict(int)
    for entry in all_entries:
        dialect_counts[entry["dialect"]] += 1
    
    print("\n   Dialect Distribution:")
    for dialect, count in sorted(dialect_counts.items(), key=lambda x: -x[1]):
        print(f"     {dialect}: {count}")
    
    # Split
    train, val, test = split_dataset(
        all_entries, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    print(f"\n📊 Split sizes: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    # Write splits
    print("\n📝 Writing dataset splits...")
    n_train = write_split(train, output_dir / "train", args.sample_rate)
    n_val = write_split(val, output_dir / "validation", args.sample_rate)
    n_test = write_split(test, output_dir / "test", args.sample_rate)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"   Train:      {n_train} samples")
    print(f"   Validation: {n_val} samples")
    print(f"   Test:       {n_test} samples")
    print(f"   Output:     {output_dir}")


if __name__ == "__main__":
    main()
