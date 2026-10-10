"""
🌾 UZHAVAN AI - Audio Preprocessing Pipeline
Handles audio cleaning, resampling, noise reduction, and VAD.

Features:
    - Resample to 16kHz mono
    - Noise reduction using spectral gating
    - Voice Activity Detection (VAD)
    - Silence trimming
    - Audio normalization
    - Duration filtering
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio preprocessing configuration."""
    sample_rate: int = 16000
    min_duration: float = 0.5  # seconds
    max_duration: float = 30.0  # seconds
    normalize: bool = True
    noise_reduce: bool = True
    trim_silence: bool = True
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    top_db: int = 30  # dB threshold for silence trimming


class AudioProcessor:
    """Complete audio preprocessing pipeline for Tamil speech."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
    
    def load_audio(self, audio_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate."""
        import librosa
        
        sr = target_sr or self.config.sample_rate
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        return audio, sr
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Peak normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave small headroom
        return audio
    
    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral gating noise reduction."""
        try:
            import noisereduce as nr
            
            # Use stationary noise reduction
            audio_clean = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,
                prop_decrease=0.75,
                n_fft=2048,
                hop_length=512,
            )
            return audio_clean
        except ImportError:
            print("  ⚠️ noisereduce not installed, skipping noise reduction")
            return audio
    
    def trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trim leading and trailing silence."""
        import librosa
        
        audio_trimmed, _ = librosa.effects.trim(
            audio, top_db=self.config.top_db
        )
        return audio_trimmed
    
    def apply_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply Voice Activity Detection to keep only speech segments."""
        try:
            import webrtcvad
            
            vad = webrtcvad.Vad(self.config.vad_aggressiveness)
            
            # webrtcvad requires 16-bit PCM at 8000, 16000, 32000, or 48000 Hz
            if sr not in (8000, 16000, 32000, 48000):
                return audio
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Process in 30ms frames
            frame_duration_ms = 30
            frame_size = int(sr * frame_duration_ms / 1000)
            frame_bytes = frame_size * 2  # 16-bit = 2 bytes per sample
            
            speech_frames = []
            for i in range(0, len(audio_bytes) - frame_bytes, frame_bytes):
                frame = audio_bytes[i:i + frame_bytes]
                if len(frame) == frame_bytes:
                    is_speech = vad.is_speech(frame, sr)
                    if is_speech:
                        frame_audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767
                        speech_frames.append(frame_audio)
            
            if speech_frames:
                return np.concatenate(speech_frames)
            return audio
            
        except ImportError:
            return audio
    
    def check_duration(self, audio: np.ndarray, sr: int) -> bool:
        """Check if audio duration is within acceptable range."""
        duration = len(audio) / sr
        return self.config.min_duration <= duration <= self.config.max_duration
    
    def get_duration(self, audio: np.ndarray, sr: int) -> float:
        """Get audio duration in seconds."""
        return len(audio) / sr
    
    def process(self, audio_path: str, apply_vad: bool = False) -> Optional[Tuple[np.ndarray, int]]:
        """
        Full preprocessing pipeline:
        1. Load and resample
        2. Noise reduction
        3. Silence trimming
        4. VAD (optional)
        5. Normalization
        6. Duration check
        """
        # Load
        audio, sr = self.load_audio(audio_path)
        
        # Noise reduction
        if self.config.noise_reduce:
            audio = self.reduce_noise(audio, sr)
        
        # Trim silence
        if self.config.trim_silence:
            audio = self.trim_silence(audio, sr)
        
        # VAD
        if apply_vad:
            audio = self.apply_vad(audio, sr)
        
        # Normalize
        if self.config.normalize:
            audio = self.normalize_audio(audio)
        
        # Check duration
        if not self.check_duration(audio, sr):
            return None
        
        return audio, sr
    
    def process_and_save(
        self,
        input_path: str,
        output_path: str,
        apply_vad: bool = False,
    ) -> Optional[float]:
        """Process audio and save to file. Returns duration or None on failure."""
        import soundfile as sf
        
        result = self.process(input_path, apply_vad=apply_vad)
        if result is None:
            return None
        
        audio, sr = result
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as 16-bit PCM WAV
        sf.write(output_path, audio, sr, subtype="PCM_16")
        
        return self.get_duration(audio, sr)
    
    def compute_features(self, audio: np.ndarray, sr: int) -> dict:
        """Compute audio features for analysis."""
        import librosa
        
        duration = len(audio) / sr
        rms = float(np.sqrt(np.mean(audio**2)))
        
        # Mel spectrogram stats
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "num_samples": len(audio),
            "rms_energy": rms,
            "max_amplitude": float(np.max(np.abs(audio))),
            "mel_mean": float(np.mean(mel_db)),
            "mel_std": float(np.std(mel_db)),
            "is_silent": rms < 0.01,
        }


def batch_process(
    input_dir: str,
    output_dir: str,
    config: Optional[AudioConfig] = None,
) -> dict:
    """Batch process all audio files in a directory."""
    from tqdm import tqdm
    
    processor = AudioProcessor(config)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(input_path.glob("**/*.wav")) + list(input_path.glob("**/*.mp3")) + list(input_path.glob("**/*.flac"))
    
    stats = {"processed": 0, "failed": 0, "skipped": 0, "total_duration": 0.0}
    
    for audio_file in tqdm(audio_files, desc="Processing audio"):
        rel_path = audio_file.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix(".wav")
        
        duration = processor.process_and_save(str(audio_file), str(out_file))
        if duration is not None:
            stats["processed"] += 1
            stats["total_duration"] += duration
        else:
            stats["skipped"] += 1
    
    return stats
