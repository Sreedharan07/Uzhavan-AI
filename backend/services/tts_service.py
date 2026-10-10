"""
🌾 UZHAVAN AI - Tamil Text-to-Speech Service
Supports: Normal mode, Slow/Elderly mode, Repeat answer.

Primary: gTTS (online)
Secondary: AI4Bharat Vakyansh (offline, if available)
"""

import os
import uuid
from pathlib import Path
from typing import Optional


TTS_OUTPUT_DIR = Path(__file__).parent.parent.parent / "temp_uploads" / "tts"


def generate_tts(
    text: str,
    mode: str = "normal",  # normal | slow | elderly
    lang: str = "ta",      # ta | en
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Generate pleasant female voice TTS audio from text.
    
    Args:
        text: Tamil or English text to speak
        mode: "normal", "slow", or "elderly" (slow + clear)
        lang: Language code ("ta" or "en")
        output_dir: Directory to save audio file
    
    Returns:
        Path to generated audio file, or None on failure
    """
    if not text.strip():
        return None
    
    out_dir = Path(output_dir) if output_dir else TTS_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
    output_path = out_dir / filename
    
    # Clean text of markdown / emojis for speech
    clean_text = text.replace("1. 🇮🇳 தமிழ்:", "").replace("2. 🇬🇧 English:", "").replace("🌱", "").replace("🧪", "").replace("🏛️", "").replace("⚠️", "").strip()
    
    # Try gTTS first (online)
    try:
        from gtts import gTTS
        
        slow = mode in ("slow", "elderly")
        tts_lang = "ta" if lang == "ta" else "en"
        
        tts = gTTS(text=clean_text[:500], lang=tts_lang, slow=slow)
        tts.save(str(output_path))
        
        return str(output_path)
    except ImportError:
        print("⚠️ gTTS not installed. Run: pip install gTTS")
    except Exception as e:
        print(f"⚠️ gTTS failed (maybe offline?): {e}")
    
    # Fallback: try pyttsx3 with female voice selection (offline)
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Select pleasant female voice if available
        voices = engine.getProperty("voices")
        female_voice = None
        for voice in voices:
            name_lower = voice.name.lower()
            if "female" in name_lower or "zira" in name_lower or "heera" in name_lower or "sabina" in name_lower or "veena" in name_lower:
                female_voice = voice.id
                break
        if female_voice:
            engine.setProperty("voice", female_voice)
        elif voices:
            engine.setProperty("voice", voices[-1].id)
        
        if mode in ("slow", "elderly"):
            engine.setProperty("rate", 110)  # Slower speed
        else:
            engine.setProperty("rate", 145)
        
        wav_path = out_dir / f"tts_{uuid.uuid4().hex[:8]}.wav"
        engine.save_to_file(clean_text[:500], str(wav_path))
        engine.runAndWait()
        
        return str(wav_path)
    except Exception:
        pass
    
    return None


def get_tts_modes():
    """Return available TTS modes."""
    return {
        "normal": {"description": "சாதாரண வேகம்", "english": "Normal speed"},
        "slow": {"description": "மெதுவான வேகம்", "english": "Slow speed"},
        "elderly": {"description": "முதியோர் பயன்முறை - மிக மெதுவாக, தெளிவாக", "english": "Elderly mode - very slow, clear"},
    }
