"""
🌾 UZHAVAN AI - Dialect-Conditioned ASR Engine
Injects dialect-specific acoustic/lexical prompts and prefix contexts into Whisper
to drastically reduce Word Error Rate (WER) on non-standard regional Tamil dialects.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ml.nlp.diglossia_normalizer import get_diglossia_normalizer


class DialectPromptManager:
    """
    Manages regional dialect acoustic priors and prompt conditioning prefixes
    for Whisper decoding.
    """

    # Dialect-specific acoustic & lexical context prefixes
    DIALECT_PROMPTS = {
        "kongu": "கொங்கு நாட்டு தமிழ் பேச்சு: நெல், கரும்பு, மஞ்சள், உரம், மானியம். போயிட்டு வர்றேனுங்க, பண்ணீட்டீங்களா.",
        "madurai": "மதுரை தென்மாவட்ட தமிழ் பேச்சு: வைகை பாசனம், உழவு, விதைப்பு, என்னய்யா, அம்புட்டுதான்.",
        "nellai": "திருநெல்வேலி நெல்லை தமிழ் பேச்சு: தாமிரபரணி பாசனம், வயல் வேலை, என்னலே, வந்தாக, போறீக.",
        "thanjavur": "தஞ்சாவூர் டெல்டா மாவட்ட தமிழ் பேச்சு: காவிரி டெல்டா, குருவை, சம்பா நெல் சாகுபடி, வெளைச்சல்.",
        "chennai": "சென்னை வடதமிழ் பேச்சு: காய்கறி மார்க்கெட் ரேட், உரம் விலை, அரசு உதவி.",
        "standard": "செந்தமிழ் உரை: வேளாண்மை, பயிர் பாதுகாப்பு, இயற்கை உரம், அரசாங்க மானியம்.",
        "code_mixed": "Tamil English code-mixed speech: paddy crop, urea fertilizer dosage, PM-KISAN scheme, subsidy.",
    }

    @classmethod
    def get_prompt_for_dialect(cls, dialect: str) -> str:
        dialect_key = dialect.lower().strip()
        if "kongu" in dialect_key:
            return cls.DIALECT_PROMPTS["kongu"]
        elif "madurai" in dialect_key:
            return cls.DIALECT_PROMPTS["madurai"]
        elif "nellai" in dialect_key or "tirunelveli" in dialect_key:
            return cls.DIALECT_PROMPTS["nellai"]
        elif "delta" in dialect_key or "thanjavur" in dialect_key:
            return cls.DIALECT_PROMPTS["thanjavur"]
        elif "chennai" in dialect_key or "madras" in dialect_key:
            return cls.DIALECT_PROMPTS["chennai"]
        elif "mix" in dialect_key or "code" in dialect_key:
            return cls.DIALECT_PROMPTS["code_mixed"]
        return cls.DIALECT_PROMPTS["standard"]


class DialectConditionedASR:
    """
    Dual-Stage Dialect-Conditioned Speech Recognizer.
    Stage 1: Front-end dialect classification (CNN-BiLSTM / Prosody).
    Stage 2: Dialect-Conditioned Whisper decoding with diglossia normalization.
    """

    def __init__(
        self,
        asr_model_dir: str = "models/asr/best_model",
        architecture: str = "whisper",
        device: Optional[str] = None,
    ):
        self.asr_model_dir = asr_model_dir
        self.architecture = architecture
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_manager = DialectPromptManager()
        self.normalizer = get_diglossia_normalizer()
        self._cached_asr = None

    def _get_asr(self):
        if self._cached_asr is None:
            from ml.models.model_factory import load_trained_model
            try:
                self._cached_asr = load_trained_model(
                    self.asr_model_dir, self.architecture, self.device
                )
            except Exception as e:
                # Graceful fallback: load small pre-trained Whisper directly if local weights aren't saved
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                model_name = "openai/whisper-small"
                processor = AutoProcessor.from_pretrained(model_name)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(self.device)
                model.eval()
                self._cached_asr = (model, processor)
        return self._cached_asr

    def transcribe_with_dialect(
        self,
        audio_path: str,
        dialect_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using dialect conditioning.

        Args:
            audio_path: Path to .wav audio file
            dialect_hint: Optional pre-detected dialect (kongu, madurai, etc.)
                          If None, will detect automatically.

        Returns:
            {
                "verbatim_transcription": str,
                "normalized_transcription": str,
                "dialect": str,
                "dialect_confidence": float,
                "latency_ms": float,
                "diglossia_metadata": dict,
                "conditioning_applied": bool
            }
        """
        import librosa

        start_time = time.time()
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Step 1: Detect dialect if not provided
        dialect_code = dialect_hint
        dialect_confidence = 1.0
        if not dialect_code:
            try:
                from ml.dialect.evaluate import classify_dialect
                dialect_result = classify_dialect(str(audio_path), device=self.device)
                dialect_code = dialect_result.get("dialect_code", "standard")
                dialect_confidence = dialect_result.get("confidence", 0.0)
            except Exception:
                dialect_code = "standard"
                dialect_confidence = 0.5

        prompt_prefix = self.prompt_manager.get_prompt_for_dialect(dialect_code)

        # Step 2: Load and preprocess audio
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        audio_duration = round(len(audio) / sr, 2)

        # Step 3: Run Dialect-Conditioned Whisper Decoding
        model, processor = self._get_asr()
        input_features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        gen_kwargs = {
            "input_features": input_features,
            "max_new_tokens": 225,
            "language": "ta",
            "task": "transcribe",
        }

        # Inject dialect prompt tokens into decoder
        try:
            prompt_ids = processor.get_prompt_ids(prompt_prefix)
            gen_kwargs["prompt_ids"] = prompt_ids
            conditioning_applied = True
        except Exception:
            # Fallback for processors without get_prompt_ids
            try:
                gen_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                    language="ta", task="transcribe"
                )
            except Exception:
                pass
            conditioning_applied = False

        with torch.no_grad():
            predicted_ids = model.generate(**gen_kwargs)

        verbatim_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        # Step 4: Pass through Diglossia Normalizer
        normalization_result = self.normalizer.normalize(verbatim_text, dialect_hint=dialect_code)
        normalized_text = normalization_result["normalized"]

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "verbatim_transcription": verbatim_text,
            "normalized_transcription": normalized_text,
            "dialect": dialect_code,
            "dialect_confidence": dialect_confidence,
            "latency_ms": latency_ms,
            "audio_duration": audio_duration,
            "conditioning_applied": conditioning_applied,
            "diglossia_metadata": normalization_result,
        }


# Singleton instance
_conditioned_asr = None

def get_dialect_conditioned_asr() -> DialectConditionedASR:
    global _conditioned_asr
    if _conditioned_asr is None:
        _conditioned_asr = DialectConditionedASR()
    return _conditioned_asr
