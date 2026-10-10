"""
🌾 UZHAVAN AI - Acoustic Scene Classification (Field Context Awareness)
MobileNetV3-Small adapted for audio — detects farmer's environment for adaptive processing.

Architecture: MobileNetV3-Small on log-Mel spectrograms
Time Complexity: O(n) linear — inverted residuals with squeeze-excite, ~10x faster than ResNet
Parameters: ~1.5M
Classes: indoor_quiet, open_field, tractor_nearby, market_crowd, pump_shed
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

SCENE_LABELS = ["indoor_quiet", "open_field", "tractor_nearby", "market_crowd", "pump_shed"]

# Adaptive noise reduction parameters per scene
SCENE_NOISE_PROFILES = {
    "indoor_quiet": {
        "noise_reduce_strength": 0.3,
        "vad_threshold": 0.01,
        "asr_beam_size": 5,
        "description": "Low noise indoor environment — minimal processing",
    },
    "open_field": {
        "noise_reduce_strength": 0.6,
        "vad_threshold": 0.03,
        "asr_beam_size": 5,
        "description": "Open field with wind — moderate wind noise suppression",
    },
    "tractor_nearby": {
        "noise_reduce_strength": 0.9,
        "vad_threshold": 0.08,
        "asr_beam_size": 8,
        "description": "Tractor engine rumble — aggressive low-frequency filtering",
    },
    "market_crowd": {
        "noise_reduce_strength": 0.7,
        "vad_threshold": 0.05,
        "asr_beam_size": 8,
        "description": "Crowded market — multi-speaker separation needed",
    },
    "pump_shed": {
        "noise_reduce_strength": 0.85,
        "vad_threshold": 0.06,
        "asr_beam_size": 8,
        "description": "Diesel pump-set — periodic mechanical noise removal",
    },
}


class AudioMobileNetV3(nn.Module):
    """
    MobileNetV3-Small adapted for audio scene classification.
    Input: log-Mel spectrogram (B, 1, 64, T) converted to pseudo-RGB (B, 3, 64, T).
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()

        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        self.features = base.features
        self.avgpool = base.avgpool

        # Custom head for scene classification
        self.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # Mono-to-RGB adapter (replicate single channel to 3)
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.channel_adapter.weight, 1.0 / 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: log-Mel spectrogram (B, 1, n_mels, T)
        Returns:
            logits (B, num_classes)
        """
        # Adapt mono mel to 3-channel pseudo-RGB
        x = self.channel_adapter(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class AcousticSceneClassifier:
    """
    End-to-end acoustic scene classifier with adaptive noise parameter output.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioMobileNetV3(num_classes=len(SCENE_LABELS))

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model = self.model.to(self.device)
        self.model.eval()

    def _compute_log_mel(self, audio: np.ndarray, sr: int = 16000, n_mels: int = 64) -> np.ndarray:
        """Compute log-Mel spectrogram from audio."""
        n_fft = 1024
        hop_length = 512
        # Simple mel spectrogram via numpy FFT
        frames = []
        for start in range(0, len(audio) - n_fft, hop_length):
            frame = audio[start:start + n_fft]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
            frames.append(spectrum)

        if not frames:
            return np.zeros((n_mels, 32), dtype=np.float32)

        spectrogram = np.array(frames).T  # (freq_bins, time)

        # Simple mel filterbank (linear approximation)
        freq_bins = spectrogram.shape[0]
        mel_fb = np.zeros((n_mels, freq_bins), dtype=np.float32)
        for i in range(n_mels):
            center = int(freq_bins * (i + 1) / (n_mels + 1))
            width = max(1, freq_bins // (n_mels * 2))
            start_bin = max(0, center - width)
            end_bin = min(freq_bins, center + width)
            mel_fb[i, start_bin:end_bin] = 1.0 / max(1, end_bin - start_bin)

        mel_spec = mel_fb @ spectrogram
        log_mel = np.log(mel_spec + 1e-10)

        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        return log_mel.astype(np.float32)

    @torch.no_grad()
    def classify_scene(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """
        Classify acoustic scene and return adaptive noise parameters.
        
        Returns:
            {
                "scene": str, "confidence": float,
                "noise_profile": dict, "all_scores": dict
            }
        """
        log_mel = self._compute_log_mel(audio, sr)

        # Ensure minimum time dimension for MobileNet
        if log_mel.shape[1] < 32:
            log_mel = np.pad(log_mel, ((0, 0), (0, 32 - log_mel.shape[1])))

        x = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float().to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)[0]

        scores = {SCENE_LABELS[i]: round(probs[i].item(), 4) for i in range(len(SCENE_LABELS))}
        top_scene = SCENE_LABELS[probs.argmax().item()]
        confidence = probs.max().item()

        return {
            "scene": top_scene,
            "confidence": round(confidence, 4),
            "noise_profile": SCENE_NOISE_PROFILES[top_scene],
            "all_scores": scores,
        }
