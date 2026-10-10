"""
🌾 UZHAVAN AI - Model Factory
Abstraction layer for multiple ASR architectures.

Supported:
    - Whisper (openai/whisper-small, etc.)
    - Wav2Vec2 (facebook/wav2vec2-large-xlsr-53, ai4bharat/indicwav2vec_v1_tamil)
    - Conformer (stub for future IndicConformer support)

Usage:
    from ml.models.model_factory import create_asr_model
    model, processor = create_asr_model("whisper", "openai/whisper-small", use_peft=True)
"""

import sys
import torch
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Reconfigure stdout/stderr for Windows UTF-8 console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def get_device_info() -> Dict[str, str]:
    """Detect and print GPU/CUDA information."""
    info = {
        "device": "cpu",
        "gpu_name": "N/A",
        "vram": "N/A",
        "cuda_version": "N/A",
        "pytorch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        info["device"] = f"cuda:{torch.cuda.current_device()}"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_mem
        info["vram"] = f"{vram_bytes / 1024**3:.1f} GB"
        info["cuda_version"] = torch.version.cuda or "N/A"
    
    return info


def print_device_info():
    """Print GPU/CUDA info to console."""
    info = get_device_info()
    print("\n🖥️  Device Information:")
    print(f"   GPU:      {info['gpu_name']}")
    print(f"   VRAM:     {info['vram']}")
    print(f"   CUDA:     {info['cuda_version']}")
    print(f"   PyTorch:  {info['pytorch_version']}")
    print(f"   Device:   {info['device']}")
    return info


def create_whisper_model(
    model_name: str = "openai/whisper-small",
    language: str = "ta",
    task: str = "transcribe",
    use_peft: bool = False,
    peft_config: Optional[Dict] = None,
) -> Tuple[Any, Any, Any]:
    """
    Create Whisper model for fine-tuning.
    
    Returns:
        (model, processor, tokenizer)
    """
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
        WhisperFeatureExtractor,
    )
    
    print(f"\n📦 Loading Whisper model: {model_name}")
    
    # Load components
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Configure for Tamil
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    
    # Apply PEFT/LoRA if requested
    if use_peft:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            peft_cfg = peft_config or {}
            lora_config = LoraConfig(
                r=peft_cfg.get("r", 16),
                lora_alpha=peft_cfg.get("lora_alpha", 32),
                lora_dropout=peft_cfg.get("lora_dropout", 0.1),
                target_modules=peft_cfg.get("target_modules", ["q_proj", "v_proj"]),
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            
            model.enable_input_require_grads()
            model = get_peft_model(model, lora_config)
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"   LoRA enabled: {trainable:,} trainable params ({100*trainable/total:.1f}%) of {total:,} total")
            
        except ImportError:
            print("   ⚠️ PEFT not installed, training full model. Run: pip install peft")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Full fine-tuning: {trainable:,} trainable parameters")
    
    return model, processor, tokenizer


def create_wav2vec2_model(
    model_name: str = "facebook/wav2vec2-large-xlsr-53",
    num_labels: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """
    Create Wav2Vec2 model for CTC-based ASR fine-tuning.
    
    Returns:
        (model, processor, tokenizer)
    """
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
    )
    
    print(f"\n📦 Loading Wav2Vec2 model: {model_name}")
    
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=vocab_size or len(processor.tokenizer),
    )
    
    # Freeze feature encoder
    model.freeze_feature_encoder()
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Feature encoder frozen: {trainable:,} trainable params of {total:,} total")
    
    return model, processor, tokenizer


def create_asr_model(
    architecture: str = "whisper",
    model_name: Optional[str] = None,
    **kwargs,
) -> Tuple[Any, Any, Any]:
    """
    Factory function to create ASR model.
    
    Args:
        architecture: "whisper" | "wav2vec2" | "conformer"
        model_name: HuggingFace model identifier
        **kwargs: Architecture-specific arguments
    
    Returns:
        (model, processor, tokenizer)
    """
    if architecture == "whisper":
        model_name = model_name or "openai/whisper-small"
        return create_whisper_model(model_name, **kwargs)
    elif architecture == "wav2vec2":
        model_name = model_name or "facebook/wav2vec2-large-xlsr-53"
        return create_wav2vec2_model(model_name, **kwargs)
    elif architecture == "conformer":
        raise NotImplementedError(
            "IndicConformer support is planned for a future release. "
            "Use 'whisper' or 'wav2vec2' for now."
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use: whisper, wav2vec2, conformer")


def load_trained_model(
    model_dir: str,
    architecture: str = "whisper",
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load a trained/fine-tuned model from checkpoint.
    
    Returns:
        (model, processor)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = Path(model_dir)
    
    if architecture == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        
        # Check for PEFT adapter
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            from peft import PeftModel
            
            # Load base model from adapter config
            import json
            with open(adapter_config) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", "openai/whisper-small")
            
            base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(base_model, model_dir)
            model = model.merge_and_unload()
        else:
            model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = model.to(device)
        model.eval()
        
        return model, processor
    
    elif architecture == "wav2vec2":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        
        model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
        processor = Wav2Vec2Processor.from_pretrained(model_dir)
        model.eval()
        
        return model, processor
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
