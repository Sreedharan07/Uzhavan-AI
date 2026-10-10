"""
🌾 UZHAVAN AI - Audio Data Augmentation
Rural-specific augmentation pipeline for Tamil speech.

Supports:
    - Background noise (farm machinery, wind, birds, rain, road, village)
    - Gaussian noise
    - Volume variation (gain)
    - Speed perturbation (time stretch)
    - Pitch variation
    - Reverberation / Room impulse response
    - Time shifting
    - Low/High pass filters

Uses: audiomentations, torchaudio, librosa
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml


class AudioAugmentor:
    """Configurable audio augmentation pipeline for rural Tamil speech."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize augmentor from config file or dict.
        
        Args:
            config_path: Path to augmentation YAML config
            config_dict: Dictionary with augmentation config
        """
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            self.config = self._default_config()
        
        self.enabled = self.config.get("enabled", True)
        self.max_transforms = self.config.get("max_transforms_per_sample", 3)
        self._pipeline = None
        self._noise_samples = {}
    
    @staticmethod
    def _default_config() -> Dict:
        """Default augmentation configuration."""
        return {
            "enabled": True,
            "max_transforms_per_sample": 3,
            "transforms": {
                "add_gaussian_noise": {
                    "enabled": True,
                    "probability": 0.3,
                    "min_amplitude": 0.001,
                    "max_amplitude": 0.015,
                },
                "gain": {
                    "enabled": True,
                    "probability": 0.4,
                    "min_gain_db": -6,
                    "max_gain_db": 6,
                },
                "time_stretch": {
                    "enabled": True,
                    "probability": 0.3,
                    "min_rate": 0.9,
                    "max_rate": 1.1,
                },
                "pitch_shift": {
                    "enabled": True,
                    "probability": 0.3,
                    "min_semitones": -2,
                    "max_semitones": 2,
                },
                "shift": {
                    "enabled": True,
                    "probability": 0.3,
                    "min_shift": -0.5,
                    "max_shift": 0.5,
                },
            },
        }
    
    def build_audiomentations_pipeline(self):
        """Build augmentation pipeline using audiomentations library."""
        try:
            from audiomentations import (
                Compose,
                AddGaussianNoise,
                Gain,
                TimeStretch,
                PitchShift,
                Shift,
                LowPassFilter,
                HighPassFilter,
                AddBackgroundNoise,
                OneOf,
            )
        except ImportError:
            print("⚠️ audiomentations not installed. Run: pip install audiomentations")
            return None
        
        transforms_config = self.config.get("transforms", {})
        transform_list = []
        
        # Gaussian noise
        cfg = transforms_config.get("add_gaussian_noise", {})
        if cfg.get("enabled", False):
            transform_list.append(
                AddGaussianNoise(
                    min_amplitude=cfg.get("min_amplitude", 0.001),
                    max_amplitude=cfg.get("max_amplitude", 0.015),
                    p=cfg.get("probability", 0.3),
                )
            )
        
        # Gain (volume variation)
        cfg = transforms_config.get("gain", {})
        if cfg.get("enabled", False):
            transform_list.append(
                Gain(
                    min_gain_db=cfg.get("min_gain_db", -6),
                    max_gain_db=cfg.get("max_gain_db", 6),
                    p=cfg.get("probability", 0.4),
                )
            )
        
        # Time stretch (speed perturbation)
        cfg = transforms_config.get("time_stretch", {})
        if cfg.get("enabled", False):
            transform_list.append(
                TimeStretch(
                    min_rate=cfg.get("min_rate", 0.9),
                    max_rate=cfg.get("max_rate", 1.1),
                    p=cfg.get("probability", 0.3),
                )
            )
        
        # Pitch shift
        cfg = transforms_config.get("pitch_shift", {})
        if cfg.get("enabled", False):
            transform_list.append(
                PitchShift(
                    min_semitones=cfg.get("min_semitones", -2),
                    max_semitones=cfg.get("max_semitones", 2),
                    p=cfg.get("probability", 0.3),
                )
            )
        
        # Time shift
        cfg = transforms_config.get("shift", {})
        if cfg.get("enabled", False):
            transform_list.append(
                Shift(
                    min_shift=cfg.get("min_shift", -0.5),
                    max_shift=cfg.get("max_shift", 0.5),
                    p=cfg.get("probability", 0.3),
                )
            )
        
        # Low-pass filter
        cfg = transforms_config.get("low_pass_filter", {})
        if cfg.get("enabled", False):
            transform_list.append(
                LowPassFilter(
                    min_cutoff_freq=cfg.get("min_cutoff", 2000),
                    max_cutoff_freq=cfg.get("max_cutoff", 7500),
                    p=cfg.get("probability", 0.2),
                )
            )
        
        # High-pass filter
        cfg = transforms_config.get("high_pass_filter", {})
        if cfg.get("enabled", False):
            transform_list.append(
                HighPassFilter(
                    min_cutoff_freq=cfg.get("min_cutoff", 50),
                    max_cutoff_freq=cfg.get("max_cutoff", 500),
                    p=cfg.get("probability", 0.2),
                )
            )
        
        # Background noise (farm machinery, wind, birds, etc.)
        cfg = transforms_config.get("add_background_noise", {})
        if cfg.get("enabled", False):
            noise_dir = cfg.get("noise_dir", "datasets/noise_samples")
            if Path(noise_dir).exists() and any(Path(noise_dir).glob("*.wav")):
                transform_list.append(
                    AddBackgroundNoise(
                        sounds_path=noise_dir,
                        min_snr_db=cfg.get("snr_db_range", [5, 20])[0],
                        max_snr_db=cfg.get("snr_db_range", [5, 20])[1],
                        p=cfg.get("probability", 0.5),
                    )
                )
        
        if not transform_list:
            return None
        
        self._pipeline = Compose(transform_list)
        return self._pipeline
    
    def augment(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Apply augmentation to audio array.
        
        Args:
            audio: Audio waveform as numpy array (float32)
            sample_rate: Sample rate of the audio
            
        Returns:
            Augmented audio array
        """
        if not self.enabled:
            return audio
        
        # Build pipeline if not already built
        if self._pipeline is None:
            self.build_audiomentations_pipeline()
        
        if self._pipeline is None:
            # Fallback to manual augmentation if audiomentations not available
            return self._manual_augment(audio, sample_rate)
        
        # Apply augmentation
        augmented = self._pipeline(samples=audio.astype(np.float32), sample_rate=sample_rate)
        return augmented
    
    def _manual_augment(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fallback augmentation using numpy/librosa when audiomentations is not available."""
        transforms_applied = 0
        audio = audio.copy()
        
        transforms_config = self.config.get("transforms", {})
        
        # Gaussian noise
        cfg = transforms_config.get("add_gaussian_noise", {})
        if cfg.get("enabled", False) and random.random() < cfg.get("probability", 0.3):
            amplitude = random.uniform(cfg.get("min_amplitude", 0.001), cfg.get("max_amplitude", 0.015))
            noise = np.random.normal(0, amplitude, len(audio))
            audio = audio + noise.astype(np.float32)
            transforms_applied += 1
        
        # Volume variation
        cfg = transforms_config.get("gain", {})
        if cfg.get("enabled", False) and random.random() < cfg.get("probability", 0.4):
            gain_db = random.uniform(cfg.get("min_gain_db", -6), cfg.get("max_gain_db", 6))
            gain_factor = 10 ** (gain_db / 20)
            audio = audio * gain_factor
            transforms_applied += 1
        
        # Speed perturbation
        cfg = transforms_config.get("time_stretch", {})
        if cfg.get("enabled", False) and random.random() < cfg.get("probability", 0.3):
            if transforms_applied < self.max_transforms:
                try:
                    import librosa
                    rate = random.uniform(cfg.get("min_rate", 0.9), cfg.get("max_rate", 1.1))
                    audio = librosa.effects.time_stretch(audio, rate=rate)
                    transforms_applied += 1
                except Exception:
                    pass
        
        # Pitch shift
        cfg = transforms_config.get("pitch_shift", {})
        if cfg.get("enabled", False) and random.random() < cfg.get("probability", 0.3):
            if transforms_applied < self.max_transforms:
                try:
                    import librosa
                    n_steps = random.uniform(cfg.get("min_semitones", -2), cfg.get("max_semitones", 2))
                    audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
                    transforms_applied += 1
                except Exception:
                    pass
        
        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio.astype(np.float32)
    
    def generate_synthetic_noise(self, output_dir: str, duration: float = 10.0, sr: int = 16000):
        """
        Generate synthetic noise samples for augmentation when real noise files aren't available.
        Creates approximations of rural environmental sounds.
        
        NOTE: These are synthetic approximations, NOT real recordings.
        For production, use actual field recordings.
        """
        import soundfile as sf
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        n_samples = int(duration * sr)
        
        noise_generators = {
            "white_noise": lambda: np.random.normal(0, 0.1, n_samples).astype(np.float32),
            "wind_approx": lambda: self._generate_wind_noise(n_samples, sr),
            "rain_approx": lambda: self._generate_rain_noise(n_samples, sr),
            "ambient_hum": lambda: self._generate_hum_noise(n_samples, sr),
        }
        
        for name, generator in noise_generators.items():
            noise = generator()
            noise = noise / (np.max(np.abs(noise)) + 1e-8) * 0.5
            filepath = output_path / f"{name}_synthetic.wav"
            sf.write(str(filepath), noise, sr, subtype="PCM_16")
            print(f"  🔊 Generated synthetic noise: {name} ({duration}s)")
        
        print(f"\n  ⚠️ These are SYNTHETIC noise approximations.")
        print(f"  For best results, collect real rural environment recordings.")
    
    @staticmethod
    def _generate_wind_noise(n_samples: int, sr: int) -> np.ndarray:
        """Generate wind-like noise using filtered noise."""
        noise = np.random.normal(0, 1, n_samples).astype(np.float32)
        # Simple low-pass approximation via moving average
        window_size = int(sr * 0.01)  # 10ms window
        kernel = np.ones(window_size) / window_size
        filtered = np.convolve(noise, kernel, mode="same")
        # Add slow amplitude modulation
        t = np.linspace(0, n_samples / sr, n_samples)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t + np.random.uniform(0, 2 * np.pi))
        return (filtered * modulation).astype(np.float32)
    
    @staticmethod
    def _generate_rain_noise(n_samples: int, sr: int) -> np.ndarray:
        """Generate rain-like noise using filtered pink noise with impulses."""
        # Pink noise base
        white = np.random.normal(0, 1, n_samples)
        # Simple pink noise approximation
        b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        from scipy.signal import lfilter
        try:
            pink = lfilter(b, a, white).astype(np.float32)
        except Exception:
            pink = white.astype(np.float32)
        # Add random impulses (rain drops)
        impulses = np.zeros(n_samples, dtype=np.float32)
        n_drops = int(n_samples * 0.001)
        drop_positions = np.random.randint(0, n_samples, n_drops)
        impulses[drop_positions] = np.random.uniform(0.1, 0.5, n_drops)
        return (pink * 0.3 + impulses * 0.7).astype(np.float32)
    
    @staticmethod
    def _generate_hum_noise(n_samples: int, sr: int) -> np.ndarray:
        """Generate electrical hum noise (50/60 Hz)."""
        t = np.linspace(0, n_samples / sr, n_samples)
        hum = 0.3 * np.sin(2 * np.pi * 50 * t)  # 50 Hz fundamental
        hum += 0.1 * np.sin(2 * np.pi * 100 * t)  # 2nd harmonic
        hum += 0.05 * np.sin(2 * np.pi * 150 * t)  # 3rd harmonic
        noise = np.random.normal(0, 0.02, n_samples)
        return (hum + noise).astype(np.float32)


def create_noise_samples(output_dir: str = "datasets/noise_samples"):
    """Create synthetic noise samples for augmentation."""
    augmentor = AudioAugmentor()
    augmentor.generate_synthetic_noise(output_dir)
