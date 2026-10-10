"""
🌾 UZHAVAN AI - ASR Training Pipeline
Fine-tune Whisper/Wav2Vec2 for Tamil Dialect Speech Recognition.

Supports:
    - Configurable epochs, batch_size, learning_rate, etc.
    - Gradient accumulation + mixed precision (AMP/FP16)
    - Checkpointing + early stopping
    - Resume training from checkpoint
    - Best model saving based on WER
    - TensorBoard logging
    - CUDA with CPU fallback

Usage:
    python ml/asr/train.py --config configs/asr.yaml
    python ml/asr/train.py --config configs/asr.yaml --epochs 5 --batch_size 4
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Any

import yaml
import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.model_factory import create_asr_model, print_device_info
from ml.asr.dataset import create_dataloaders, ASRDataCollator
from ml.preprocessing.augmentation import AudioAugmentor


def compute_wer_cer(predictions: list, references: list) -> Dict[str, float]:
    """Compute Word Error Rate and Character Error Rate."""
    try:
        from jiwer import wer, cer
        
        # Filter out empty references
        valid_pairs = [
            (p, r) for p, r in zip(predictions, references)
            if r.strip()
        ]
        
        if not valid_pairs:
            return {"wer": 1.0, "cer": 1.0}
        
        preds, refs = zip(*valid_pairs)
        preds = list(preds)
        refs = list(refs)
        
        return {
            "wer": wer(refs, preds),
            "cer": cer(refs, preds),
        }
    except ImportError:
        print("⚠️ jiwer not installed. Run: pip install jiwer")
        return {"wer": -1.0, "cer": -1.0}


class ASRTrainer:
    """
    Complete ASR training pipeline with all features:
    epochs, checkpointing, early stopping, mixed precision,
    gradient accumulation, TensorBoard, and WER evaluation.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config.get("model", {})
        self.train_config = config.get("training", {})
        self.data_config = config.get("data", {})
        self.eval_config = config.get("evaluation", {})
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_device_info()
        
        # Training hyperparameters
        self.epochs = self.train_config.get("epochs", 30)
        self.lr = self.train_config.get("learning_rate", 3e-5)
        self.warmup_steps = self.train_config.get("warmup_steps", 500)
        self.weight_decay = self.train_config.get("weight_decay", 0.01)
        self.grad_accum_steps = self.train_config.get("gradient_accumulation_steps", 4)
        self.mixed_precision = self.train_config.get("mixed_precision", True) and self.device.type == "cuda"
        self.max_grad_norm = self.train_config.get("max_grad_norm", 1.0)
        
        # Checkpointing
        self.save_best = self.train_config.get("save_best_model", True)
        self.checkpoint_dir = Path(self.train_config.get("checkpoint_dir", "models/asr/checkpoints"))
        self.best_model_dir = Path(self.train_config.get("best_model_dir", "models/asr/best_model"))
        self.save_every_n = self.train_config.get("save_every_n_epochs", 5)
        
        # Early stopping
        self.early_stopping = self.train_config.get("early_stopping", True)
        self.patience = self.train_config.get("early_stopping_patience", 5)
        self.es_metric = self.train_config.get("early_stopping_metric", "wer")
        self.es_mode = self.train_config.get("early_stopping_mode", "min")
        
        # TensorBoard
        log_dir = self.train_config.get("log_dir", "results/asr/tensorboard")
        self.writer = SummaryWriter(log_dir)
        
        # State
        self.global_step = 0
        self.best_metric = float("inf") if self.es_mode == "min" else float("-inf")
        self.patience_counter = 0
        self.start_epoch = 0
        
        # Model components (initialized in setup)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
    
    def setup(self):
        """Initialize model, optimizer, scheduler, and data."""
        # Create model
        architecture = self.model_config.get("architecture", "whisper")
        model_name = self.model_config.get("name", "openai/whisper-small")
        use_peft = self.model_config.get("use_peft", False)
        peft_config = self.model_config.get("peft", {})
        
        self.model, self.processor, self.tokenizer = create_asr_model(
            architecture=architecture,
            model_name=model_name,
            language=self.model_config.get("language", "ta"),
            task=self.model_config.get("task", "transcribe"),
            use_peft=use_peft,
            peft_config=peft_config,
        )
        self.model = self.model.to(self.device)
        
        # Augmentor
        augmentor = None
        aug_config = self.data_config.get("augmentation_config")
        if self.data_config.get("augment_train", False) and aug_config:
            augmentor = AudioAugmentor(config_path=aug_config)
        
        # Data loaders
        self.dataloaders = create_dataloaders(
            self.config, self.processor, augmentor=augmentor
        )
        
        if "train" not in self.dataloaders:
            print("\n❌ No training data found! Run dataset preparation first:")
            print("   python scripts/download_datasets.py --dataset openslr_65")
            print("   python scripts/prepare_dataset.py --source all")
            return False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(self.dataloaders["train"]) * self.epochs // self.grad_accum_steps
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - self.warmup_steps)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision scaler
        if self.mixed_precision:
            self.scaler = GradScaler()
            print("   ⚡ Mixed precision (FP16) enabled")
        
        # Resume from checkpoint
        resume_path = self.train_config.get("resume_from")
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(resume_path)
        
        return True
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_loader = self.dataloaders["train"]
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with optional mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(input_features=input_features, labels=labels)
                    loss = outputs.loss / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_features=input_features, labels=labels)
                loss = outputs.loss / self.grad_accum_steps
                loss.backward()
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log to TensorBoard
                log_every = self.train_config.get("log_every_n_steps", 10)
                if self.global_step % log_every == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/loss", loss.item() * self.grad_accum_steps, self.global_step)
                    self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)
                    
                    if self.device.type == "cuda":
                        gpu_mem = torch.cuda.memory_allocated() / 1024**3
                        self.writer.add_scalar("system/gpu_memory_gb", gpu_mem, self.global_step)
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
        
        return {"train_loss": total_loss / max(1, num_batches)}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation and compute metrics."""
        if "val" not in self.dataloaders:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_references = []
        
        val_loader = self.dataloaders["val"]
        
        for batch in tqdm(val_loader, desc="Validating"):
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Compute loss
            outputs = self.model(input_features=input_features, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Generate predictions for WER
            predicted_ids = self.model.generate(input_features=input_features, max_new_tokens=225)
            predictions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            # Decode references (replace -100 with pad token)
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = self.processor.tokenizer.pad_token_id
            references = self.processor.batch_decode(labels_for_decode, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
        
        # Compute metrics
        metrics = compute_wer_cer(all_predictions, all_references)
        metrics["val_loss"] = total_loss / max(1, num_batches)
        
        # Log to TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        save_dir = self.best_model_dir if is_best else self.checkpoint_dir / f"epoch_{epoch+1}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(save_dir))
        self.processor.save_pretrained(str(save_dir))
        
        # Save training state
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }
        torch.save(state, save_dir / "training_state.pt")
        
        label = "🏆 BEST" if is_best else "💾"
        print(f"  {label} Checkpoint saved to: {save_dir}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        path = Path(checkpoint_path)
        state_file = path / "training_state.pt"
        
        if state_file.exists():
            state = torch.load(state_file, map_location=self.device)
            self.start_epoch = state["epoch"] + 1
            self.global_step = state["global_step"]
            self.best_metric = state["best_metric"]
            self.patience_counter = state["patience_counter"]
            
            if "optimizer_state_dict" in state:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            
            print(f"  ▶️ Resumed from epoch {self.start_epoch}, step {self.global_step}")
    
    def _check_early_stopping(self, metric_value: float) -> bool:
        """Check if training should stop early. Returns True if should stop."""
        improved = False
        if self.es_mode == "min":
            improved = metric_value < self.best_metric
        else:
            improved = metric_value > self.best_metric
        
        if improved:
            self.best_metric = metric_value
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"\n  ⏹️ Early stopping! No improvement for {self.patience} epochs.")
                return True
            return False
    
    def train(self):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("🌾 UZHAVAN AI - ASR Training")
        print("=" * 60)
        
        if not self.setup():
            return
        
        print(f"\n📋 Training Configuration:")
        print(f"   Epochs:        {self.epochs}")
        print(f"   Batch size:    {self.train_config.get('batch_size', 8)}")
        print(f"   Learning rate: {self.lr}")
        print(f"   Warmup steps:  {self.warmup_steps}")
        print(f"   Grad accum:    {self.grad_accum_steps}")
        print(f"   Mixed prec:    {self.mixed_precision}")
        print(f"   Early stop:    {self.early_stopping} (patience={self.patience})")
        print(f"   Train samples: {len(self.dataloaders['train'].dataset)}")
        if "val" in self.dataloaders:
            print(f"   Val samples:   {len(self.dataloaders['val'].dataset)}")
        
        training_log = []
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = {}
            eval_every = self.train_config.get("eval_every_n_epochs", 1)
            if epoch % eval_every == 0 or epoch == self.epochs - 1:
                val_metrics = self.validate(epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, "epoch_time": epoch_time}
            training_log.append({"epoch": epoch + 1, **metrics})
            
            # Print epoch summary
            summary = f"\n📊 Epoch {epoch+1}/{self.epochs} ({epoch_time:.1f}s)"
            summary += f" | Loss: {train_metrics['train_loss']:.4f}"
            if "val_loss" in val_metrics:
                summary += f" | Val Loss: {val_metrics['val_loss']:.4f}"
            if "wer" in val_metrics:
                summary += f" | WER: {val_metrics['wer']:.4f}"
            if "cer" in val_metrics:
                summary += f" | CER: {val_metrics['cer']:.4f}"
            print(summary)
            
            self.writer.add_scalar("time/epoch_seconds", epoch_time, epoch)
            
            # Check for improvement and save
            is_best = False
            if val_metrics and self.es_metric in val_metrics:
                metric_val = val_metrics[self.es_metric]
                
                # Check early stopping
                should_stop = False
                if self.early_stopping:
                    should_stop = self._check_early_stopping(metric_val)
                
                # Save best model
                if self.save_best and self.patience_counter == 0:
                    is_best = True
                    self._save_checkpoint(epoch, metrics, is_best=True)
                
                if should_stop:
                    break
            
            # Periodic checkpoint
            if (epoch + 1) % self.save_every_n == 0 and not is_best:
                self._save_checkpoint(epoch, metrics, is_best=False)
        
        total_time = time.time() - start_time
        
        # Final summary
        print("\n" + "=" * 60)
        print("🏁 Training Complete!")
        print(f"   Total time:   {total_time/3600:.2f} hours")
        print(f"   Best {self.es_metric}: {self.best_metric:.4f}")
        print(f"   Best model:   {self.best_model_dir}")
        
        # Save training log
        results_dir = Path(self.eval_config.get("results_dir", "results/asr"))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = results_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"   Training log: {log_path}")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="🌾 UZHAVAN AI - Train ASR Model")
    parser.add_argument("--config", type=str, default="configs/asr.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--no-peft", action="store_true", help="Disable PEFT/LoRA")
    args = parser.parse_args()
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.resume:
        config["training"]["resume_from"] = args.resume
    if args.no_peft:
        config["model"]["use_peft"] = False
    
    # Train
    trainer = ASRTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
