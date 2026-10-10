"""
🌾 UZHAVAN AI - Dialect Classifier Training
Trains CNN+BiLSTM or Wav2Vec2+Classifier for Tamil dialect detection.

Usage:
    python ml/dialect/train.py --config configs/dialect.yaml
    python ml/dialect/train.py --config configs/dialect.yaml --architecture wav2vec_classifier
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ml.dialect.models import create_dialect_model


class DialectDataset(Dataset):
    """Dataset for dialect classification from mel spectrograms."""
    
    def __init__(
        self,
        manifest_path: str,
        dialect_labels: List[str],
        sample_rate: int = 16000,
        segment_duration: float = 5.0,
        n_mels: int = 80,
        is_training: bool = True,
    ):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sample_rate)
        self.n_mels = n_mels
        self.is_training = is_training
        self.dialect_to_idx = {d: i for i, d in enumerate(dialect_labels)}
        
        manifest = Path(manifest_path)
        audio_dir = manifest.parent / "audio"
        
        self.samples = []
        if manifest.exists():
            with open(manifest, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dialect = row.get("dialect", "standard")
                    if dialect in self.dialect_to_idx:
                        self.samples.append({
                            "audio_path": str(audio_dir / row["audio_path"]),
                            "dialect": dialect,
                            "label": self.dialect_to_idx[dialect],
                        })
        
        print(f"  📂 Loaded {len(self.samples)} dialect samples from {manifest_path}")
        
        # Print distribution
        dist = defaultdict(int)
        for s in self.samples:
            dist[s["dialect"]] += 1
        for d, c in sorted(dist.items()):
            print(f"     {d}: {c}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        import librosa
        
        sample = self.samples[idx]
        
        try:
            audio, sr = librosa.load(sample["audio_path"], sr=self.sample_rate, mono=True)
        except Exception:
            audio = np.zeros(self.segment_samples, dtype=np.float32)
        
        # Fixed-length segment
        if len(audio) > self.segment_samples:
            if self.is_training:
                start = np.random.randint(0, len(audio) - self.segment_samples)
            else:
                start = 0
            audio = audio[start:start + self.segment_samples]
        else:
            audio = np.pad(audio, (0, self.segment_samples - len(audio)))
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels, n_fft=1024, hop_length=512
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        
        # Add channel dimension: (1, n_mels, time)
        mel_tensor = torch.FloatTensor(mel_db).unsqueeze(0)
        
        return {
            "input": mel_tensor,
            "label": torch.LongTensor([sample["label"]])[0],
            "dialect": sample["dialect"],
        }


class DialectTrainer:
    """Complete dialect classifier training pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config.get("model", {})
        self.train_config = config.get("training", {})
        self.data_config = config.get("data", {})
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dialect_labels = config.get("dialects", [
            "standard", "kongu", "madurai", "tirunelveli", "nellai",
            "thanjavur", "chennai", "northern", "southern", "code_mixed",
        ])
        
        self.epochs = self.train_config.get("epochs", 50)
        self.lr = self.train_config.get("learning_rate", 1e-4)
        self.mixed_precision = self.train_config.get("mixed_precision", True) and self.device.type == "cuda"
        
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.patience = self.train_config.get("early_stopping_patience", 10)
    
    def setup(self):
        """Initialize model, data, optimizer."""
        architecture = self.model_config.get("architecture", "cnn_bilstm")
        
        print(f"\n📦 Creating dialect model: {architecture}")
        self.model = create_dialect_model(architecture, self.model_config)
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Parameters: {trainable:,} trainable / {total_params:,} total")
        
        # Datasets
        segment_dur = self.data_config.get("segment_duration", 5.0)
        n_mels = self.model_config.get("cnn_bilstm", {}).get("n_mels", 80)
        
        train_dataset = DialectDataset(
            self.data_config.get("train_manifest", "datasets/tamil_rural/train/metadata.csv"),
            self.dialect_labels,
            segment_duration=segment_dur,
            n_mels=n_mels,
            is_training=True,
        )
        val_dataset = DialectDataset(
            self.data_config.get("val_manifest", "datasets/tamil_rural/validation/metadata.csv"),
            self.dialect_labels,
            segment_duration=segment_dur,
            n_mels=n_mels,
            is_training=False,
        )
        
        batch_size = self.train_config.get("batch_size", 32)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        if len(train_dataset) == 0:
            print("\n❌ No training data found!")
            return False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.train_config.get("weight_decay", 0.01))
        
        # Scheduler
        if self.train_config.get("scheduler") == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # TensorBoard
        log_dir = self.train_config.get("log_dir", "results/dialect/tensorboard")
        self.writer = SummaryWriter(log_dir)
        
        return True
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
            inputs = batch["input"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / max(1, len(self.train_loader))
    
    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0
        
        for batch in self.val_loader:
            inputs = batch["input"].to(self.device)
            labels = batch["label"].to(self.device)
            
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        if not all_labels:
            return {"val_loss": 0, "accuracy": 0, "f1_macro": 0}
        
        return {
            "val_loss": total_loss / max(1, len(self.val_loader)),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        }
    
    def train(self):
        print("\n" + "=" * 60)
        print("🌾 UZHAVAN AI - Dialect Classifier Training")
        print("=" * 60)
        
        if not self.setup():
            return
        
        best_model_path = Path(self.train_config.get("best_model_path", "models/dialect/best_dialect_model.pt"))
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()
            
            # Log
            self.writer.add_scalar("train/loss", train_loss, epoch)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"val/{k}", v, epoch)
            
            f1 = val_metrics.get("f1_macro", 0)
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f} | Val F1={f1:.4f} | Acc={val_metrics.get('accuracy', 0):.4f}")
            
            # Save best
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "dialect_labels": self.dialect_labels,
                    "architecture": self.model_config.get("architecture", "cnn_bilstm"),
                    "config": self.model_config,
                    "metrics": val_metrics,
                    "epoch": epoch,
                }, str(best_model_path))
                print(f"  🏆 Best model saved! F1={f1:.4f}")
            else:
                self.patience_counter += 1
                if self.train_config.get("early_stopping", True) and self.patience_counter >= self.patience:
                    print(f"  ⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        self.writer.close()
        print(f"\n✅ Training complete. Best F1: {self.best_f1:.4f}")
        print(f"   Best model: {best_model_path}")


def main():
    parser = argparse.ArgumentParser(description="🌾 Train Dialect Classifier")
    parser.add_argument("--config", type=str, default="configs/dialect.yaml")
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if args.architecture:
        config["model"]["architecture"] = args.architecture
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    
    trainer = DialectTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
