"""
🌾 UZHAVAN AI - ASR Dataset
PyTorch Dataset for Tamil speech recognition training.

Handles:
    - Loading audio + transcripts from metadata CSV
    - Feature extraction using WhisperProcessor
    - On-the-fly audio augmentation during training
    - Proper batching with data collator
"""

import csv
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from torch.utils.data import Dataset, DataLoader

import yaml


class TamilASRDataset(Dataset):
    """
    PyTorch Dataset for Tamil ASR training.
    
    Loads audio and transcripts from metadata CSV file.
    Applies preprocessing and optional augmentation.
    """
    
    def __init__(
        self,
        manifest_path: str,
        processor: Any,
        audio_dir: Optional[str] = None,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        augmentor: Any = None,
        is_training: bool = True,
    ):
        """
        Args:
            manifest_path: Path to metadata.csv
            processor: WhisperProcessor or Wav2Vec2Processor
            audio_dir: Directory containing audio files (if paths in CSV are relative)
            sample_rate: Target sample rate
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            augmentor: AudioAugmentor instance for training augmentation
            is_training: Whether this is a training dataset (enables augmentation)
        """
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.augmentor = augmentor if is_training else None
        self.is_training = is_training
        
        manifest = Path(manifest_path)
        if audio_dir:
            self.audio_dir = Path(audio_dir)
        else:
            self.audio_dir = manifest.parent / "audio"
        
        # Load metadata
        self.samples = []
        if manifest.exists():
            with open(manifest, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Filter by duration
                    duration = float(row.get("duration", 0))
                    if duration > 0 and (duration < min_duration or duration > max_duration):
                        continue
                    
                    # Must have transcript
                    transcript = row.get("transcript", "").strip()
                    if not transcript:
                        continue
                    
                    self.samples.append({
                        "audio_path": str(self.audio_dir / row["audio_path"]),
                        "transcript": transcript,
                        "dialect": row.get("dialect", "standard"),
                        "speaker_id": row.get("speaker_id", ""),
                        "domain": row.get("domain", "general"),
                    })
        
        print(f"  📂 Loaded {len(self.samples)} samples from {manifest_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load audio
        import librosa
        try:
            audio, sr = librosa.load(sample["audio_path"], sr=self.sample_rate, mono=True)
        except Exception as e:
            # Return a silent audio on error (will be filtered by collator)
            audio = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence
            sr = self.sample_rate
        
        # Apply augmentation during training
        if self.augmentor is not None and self.is_training:
            audio = self.augmentor.augment(audio, sr)
        
        # Extract features using processor
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features[0]
        
        # Tokenize transcript
        labels = self.processor.tokenizer(
            sample["transcript"],
            return_tensors="pt",
        ).input_ids[0]
        
        return {
            "input_features": input_features,
            "labels": labels,
            "dialect": sample["dialect"],
            "transcript": sample["transcript"],
        }


class ASRDataCollator:
    """
    Data collator for Whisper ASR training.
    Handles variable-length labels with proper padding.
    """
    
    def __init__(self, processor: Any, decoder_start_token_id: Optional[int] = None):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack input features
        input_features = torch.stack([f["input_features"] for f in features])
        
        # Pad labels
        label_features = [f["labels"] for f in features]
        max_label_length = max(len(l) for l in label_features)
        
        labels = torch.full(
            (len(label_features), max_label_length),
            fill_value=-100,  # Ignored by loss
            dtype=torch.long,
        )
        for i, label in enumerate(label_features):
            labels[i, :len(label)] = label
        
        return {
            "input_features": input_features,
            "labels": labels,
        }


def create_dataloaders(
    config: Dict,
    processor: Any,
    augmentor: Any = None,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders from config.
    
    Args:
        config: Configuration dictionary (from asr.yaml)
        processor: WhisperProcessor or Wav2Vec2Processor
        augmentor: AudioAugmentor for training augmentation
    
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    
    sample_rate = data_cfg.get("sample_rate", 16000)
    max_duration = data_cfg.get("max_duration_seconds", 30)
    min_duration = data_cfg.get("min_duration_seconds", 0.5)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)
    persistent_workers = data_cfg.get("persistent_workers", True) and num_workers > 0
    
    batch_size = training_cfg.get("batch_size", 8)
    eval_batch_size = training_cfg.get("eval_batch_size", batch_size)
    
    # Data collator
    collator = ASRDataCollator(processor)
    
    loaders = {}
    
    # Training
    train_manifest = data_cfg.get("train_manifest")
    if train_manifest and Path(train_manifest).exists():
        train_dataset = TamilASRDataset(
            manifest_path=train_manifest,
            processor=processor,
            sample_rate=sample_rate,
            max_duration=max_duration,
            min_duration=min_duration,
            augmentor=augmentor,
            is_training=True,
        )
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collator,
            drop_last=True,
        )
    
    # Validation
    val_manifest = data_cfg.get("val_manifest")
    if val_manifest and Path(val_manifest).exists():
        val_dataset = TamilASRDataset(
            manifest_path=val_manifest,
            processor=processor,
            sample_rate=sample_rate,
            max_duration=max_duration,
            min_duration=min_duration,
            augmentor=None,
            is_training=False,
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collator,
        )
    
    # Test
    test_manifest = data_cfg.get("test_manifest")
    if test_manifest and Path(test_manifest).exists():
        test_dataset = TamilASRDataset(
            manifest_path=test_manifest,
            processor=processor,
            sample_rate=sample_rate,
            max_duration=max_duration,
            min_duration=min_duration,
            augmentor=None,
            is_training=False,
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collator,
        )
    
    return loaders
