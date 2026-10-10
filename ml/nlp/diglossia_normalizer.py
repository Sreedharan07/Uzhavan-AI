"""
🌾 UZHAVAN AI - Tamil Diglossia Normalizer
Handles the diglossic gap between spoken rural colloquial Tamil (Kodumtamil)
and standard written Tamil (Centamil).

Features:
1. Dialect morphological transformation (Kongu, Madurai, Nellai, Delta, Chennai)
2. Agricultural slang & colloquial pest/crop terminology normalization
3. Traditional rural measurement units normalization (e.g. குழி, செண்டு, மா, படி)
4. Dual-Head output: (a) Verbatim Dialect Text, (b) Normalized Standard Tamil
"""

import re
from typing import Dict, List, Tuple, Any, Optional

# Dialect-specific morphological markers & verbal suffixes
DIALECT_MORPHOLOGY = {
    "kongu": [
        (r"போயிட்டு\s+வர்றேனுங்க", "போய் வருகிறேன்"),
        (r"வர்றேனுங்க", "வருகிறேன்"),
        (r"போறேனுங்க", "போகிறேன்"),
        (r"செஞ்சேனுங்க", "செய்தேன்"),
        (r"பண்ணீட்டீங்களா", "செய்துவிட்டீர்களா"),
        (r"இருக்கீங்களா", "இருக்கிறீர்களா"),
        (r"போடணுங்க", "இட வேண்டும்"),
        (r"போடனும்ங்க", "இட வேண்டும்"),
        (r"செய்யணுங்க", "செய்ய வேண்டும்"),
        (r"பண்ணணுங்க", "செய்ய வேண்டும்"),
        (r"விடணுங்க", "பாய்ச்ச வேண்டும்"),
        (r"ஆயிடுச்சுங்க", "ஆகிவிட்டது"),
        (r"சொல்லுங்கண்ணா", "சொல்லுங்கள்"),
        (r"சொல்றேனுங்க", "சொல்கிறேன்"),
        (r"போட்டேனுங்க", "போட்டேன்"),
        (r"வேணுமுங்க", "வேண்டும்"),
        (r"ஆகுமுங்க", "ஆகும்"),
        (r"இல்லீங்க", "இல்லை"),
        (r"ஆமாங்க", "ஆம்"),
        (r"எங்கன", "எங்கே"),
        (r"அங்கன", "அங்கே"),
        (r"இங்கன", "இங்கே"),
        (r"வாங்கிட்டு\s+வந்தேனுங்க", "வாங்கி வந்தேன்"),
    ],
    "nellai": [
        (r"எங்கிட்டு", "எங்கே"),
        (r"அங்கிட்டு", "அங்கே"),
        (r"இங்கிட்டு", "இங்கே"),
        (r"செஞ்சாக", "செய்தார்கள்"),
        (r"வந்தாக", "வந்தார்கள்"),
        (r"போனாக", "போனார்கள்"),
        (r"சொன்னாக", "சொன்னார்கள்"),
        (r"இருக்காக", "இருக்கிறார்கள்"),
        (r"போறீகளா", "போகிறீர்களா"),
        (r"வாறீகளா", "வருகிறீர்களா"),
        (r"என்னலே", "என்ன"),
        (r"போலே", "போ"),
        (r"வாலே", "வா"),
        (r"செய்லே", "செய்"),
        (r"பாப்போம்லே", "பார்ப்போம்"),
        (r"மருவாதையா", "மரியாதையாக"),
        (r"சும்மாரு", "சும்மா இரு"),
    ],
    "madurai": [
        (r"செஞ்சாய்ங்களா", "செய்தீர்களா"),
        (r"போய்ட்டாய்ங்களா", "போய்விட்டார்களா"),
        (r"வந்தாய்ங்களா", "வந்தார்களா"),
        (r"அம்புட்டுதான்", "அவ்வளவுதான்"),
        (r"இம்புட்டுதான்", "இவ்வளவுதான்"),
        (r"எம்புட்டு", "எவ்வளவு"),
        (r"என்னய்யா", "என்ன"),
        (r"சொல்லுய்யா", "சொல்லுங்கள்"),
        (r"கெடக்கு", "இருக்கிறது"),
        (r"மொறைக்காத", "பார்க்காதே"),
        (r"கையிலெடுத்து", "எடுத்து"),
        (r"பாத்துக்கலாம்", "பார்த்துக் கொள்ளலாம்"),
    ],
    "thanjavur": [
        (r"சொல்றாக", "சொல்கிறார்கள்"),
        (r"செய்கிறாக", "செய்கிறார்கள்"),
        (r"வர்ராக", "வருகிறார்கள்"),
        (r"கெழமை", "கிழமை"),
        (r"வெளைச்சல்", "விளைச்சல்"),
        (r"கொலவெறி", "அதிகமாக"),
        (r"நாத்து\s+நடவு", "நாற்று நடவு"),
        (r"தண்ணி\s+பாய்ச்சியாச்சு", "நீர் பாசனம் செய்யப்பட்டது"),
    ],
    "chennai": [
        (r"துட்டு", "பணம்"),
        (r"காயிதா", "காகிதம்"),
        (r"பீட்டரு", "ஆங்கிலம்"),
        (r"மஸ்தா", "அழகாக"),
        (r"கெளம்பு", "புறப்படு"),
        (r"நைனா", "அப்பா"),
        (r"காசு\s+போயிடுச்சி", "பணம் செலவாகிவிட்டது"),
    ],
}

# Common colloquial verbal contractions across rural spoken Tamil
COLLOQUIAL_CONTRACTIONS = [
    (r"\bபண்ணனும்\b", "செய்ய வேண்டும்"),
    (r"\bபண்ணணும்\b", "செய்ய வேண்டும்"),
    (r"\bசெய்யனும்\b", "செய்ய வேண்டும்"),
    (r"\bசெய்யணும்\b", "செய்ய வேண்டும்"),
    (r"\bபோடணும்\b", "இட வேண்டும்"),
    (r"\bபோடனும்\b", "இட வேண்டும்"),
    (r"\bவிடணும்\b", "பாய்ச்ச வேண்டும்"),
    (r"\bவிடனும்\b", "பாய்ச்ச வேண்டும்"),
    (r"\bசொல்லுங்க\b", "கூறுங்கள்"),
    (r"\bதெரியல\b", "தெரியவில்லை"),
    (r"\bதெரியலை\b", "தெரியவில்லை"),
    (r"\bவரல\b", "வரவில்லை"),
    (r"\bபோகல\b", "போகவில்லை"),
    (r"\bஆயிடுச்சு\b", "ஆகிவிட்டது"),
    (r"\bஆகிடுச்சு\b", "ஆகிவிட்டது"),
    (r"\bஇருக்குது\b", "இருக்கிறது"),
    (r"\bஇருக்குப்பா\b", "இருக்கிறது"),
    (r"\bபோயிடுச்சு\b", "போய்விட்டது"),
    (r"\bஇன்னா\b", "என்ன"),
    (r"\bதண்ணி\b", "நீர்"),
    (r"\bகாட்டிலும்\b", "விட"),
    (r"\bஇன்னிக்கு\b", "இன்று"),
    (r"\bஇன்னைக்கு\b", "இன்று"),
    (r"\bநாளைக்கு\b", "நாளை"),
]

# Agronomic rural colloquialisms to canonical agricultural terms
AGRI_COLLOQUIAL_THESAURUS = {
    "செம்பேன்": {"canonical": "சிவப்பு சிலந்திப் பூச்சி", "english": "red spider mite", "category": "pest"},
    "கரிச்சட்டான்": {"canonical": "கருப்பு தண்டு வண்டு", "english": "black stem borer", "category": "pest"},
    "சுருட்டல்": {"canonical": "இலைச் சுருள் நோய்", "english": "leaf curl disease", "category": "disease"},
    "சுருட்டுப் புழு": {"canonical": "இலை சுருட்டுப் புழு", "english": "leaf folder caterpillar", "category": "pest"},
    "சாம்பல் நோய்": {"canonical": "சாம்பல் பூஞ்சை நோய்", "english": "powdery mildew", "category": "disease"},
    "செம்புள்ளி": {"canonical": "இலைப்புள்ளி நோய்", "english": "cercospora leaf spot", "category": "disease"},
    "வாடல்": {"canonical": "பாக்டீரியா வாடல் நோய்", "english": "bacterial wilt", "category": "disease"},
    "அழுகல்": {"canonical": "தண்டு மற்றும் வேர் அழுகல் நோய்", "english": "stem and root rot", "category": "disease"},
    "சாணி உரம்": {"canonical": "தொழு உரம்", "english": "farmyard manure", "category": "fertilizer"},
    "குப்பை உரம்": {"canonical": "மட்கிய இயற்கை உரம்", "english": "compost manure", "category": "fertilizer"},
    "வெள்ளை பூச்சி": {"canonical": "வெள்ளை ஈ", "english": "whitefly", "category": "pest"},
    "பச்சை புழு": {"canonical": "காய்த்துளைப்பான் புழு", "english": "pod borer caterpillar", "category": "pest"},
    "வெட்டுக்கிளி": {"canonical": "வெட்டுக்கிளி பூச்சி", "english": "grasshopper / locust", "category": "pest"},
    "ஆனைக்கொம்பன்": {"canonical": "நெல் தண்டு ஈ", "english": "rice gall midge", "category": "pest"},
    "குலை நோய்": {"canonical": "நெல் குலை நோய்", "english": "rice blast", "category": "disease"},
    "செம்பழுப்பு": {"canonical": "பழுப்பு இலைப்புள்ளி", "english": "brown spot", "category": "disease"},
}

# Traditional Tamil agrarian measurement units and conversions
TRADITIONAL_UNITS = {
    "குழி": {"metric": "0.33 செண்ட் (144 சதுர அடி)", "acres": 0.0033},
    "செண்டு": {"metric": "1 செண்ட் (435.6 சதுர அடி)", "acres": 0.01},
    "மா": {"metric": "100 குழி (33.3 செண்ட்)", "acres": 0.333},
    "காணி": {"metric": "1.32 ஏக்கர் (4 காணி = 5.28 ஏக்கர்)", "acres": 1.32},
    "மரக்கால்": {"metric": "சுமார் 4.5 கிலோகிராம்", "standard": "4.5 kg"},
    "படி": {"metric": "சுமார் 1.5 கிலோகிராம்", "standard": "1.5 kg"},
    "மூட்டை": {"metric": "சுமார் 60-75 கிலோகிராம்", "standard": "75 kg"},
    "குண்டு": {"metric": "1 குவிண்டால் (100 கிலோகிராம்)", "standard": "100 kg"},
}


class DiglossiaNormalizer:
    """
    Normalizes rural spoken Tamil (Kodumtamil) to standard literary Tamil (Centamil)
    while capturing linguistic dialect cues for research analysis.
    """

    def __init__(self):
        self.morphology_rules = DIALECT_MORPHOLOGY
        self.contractions = COLLOQUIAL_CONTRACTIONS
        self.thesaurus = AGRI_COLLOQUIAL_THESAURUS
        self.traditional_units = TRADITIONAL_UNITS

    def normalize(self, text: str, dialect_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Normalize colloquial Tamil text into standard written Tamil.

        Args:
            text: Spoken/colloquial Tamil sentence
            dialect_hint: Optional prior dialect label (kongu, madurai, etc.)

        Returns:
            {
                "verbatim": str,
                "normalized": str,
                "detected_dialect_cues": List[str],
                "inferred_dialect": Optional[str],
                "terms_mapped": List[Dict[str, str]],
                "units_detected": List[Dict[str, Any]],
                "diglossia_distance": float,  # character-level edit distance ratio
            }
        """
        if not text:
            return {
                "verbatim": "",
                "normalized": "",
                "detected_dialect_cues": [],
                "inferred_dialect": "standard",
                "terms_mapped": [],
                "units_detected": [],
                "diglossia_distance": 0.0,
            }

        verbatim = text.strip()
        normalized = verbatim
        detected_cues = []
        dialect_scores: Dict[str, int] = {k: 0 for k in self.morphology_rules.keys()}

        # 1. Apply dialect-specific morphology transformations
        dialects_to_check = [dialect_hint] if (dialect_hint and dialect_hint in self.morphology_rules) else list(self.morphology_rules.keys())
        
        for dialect in dialects_to_check:
            for pattern, replacement in self.morphology_rules[dialect]:
                matches = re.findall(pattern, normalized)
                if matches:
                    dialect_scores[dialect] += len(matches)
                    detected_cues.extend(matches)
                    normalized = re.sub(pattern, replacement, normalized)

        # 2. Apply general colloquial contractions
        for pattern, replacement in self.contractions:
            normalized = re.sub(pattern, replacement, normalized)

        # 3. Map agronomic slang terms
        terms_mapped = []
        for slang, info in self.thesaurus.items():
            if slang in normalized:
                normalized = normalized.replace(slang, info["canonical"])
                terms_mapped.append({
                    "colloquial": slang,
                    "canonical": info["canonical"],
                    "english": info["english"],
                    "category": info["category"],
                })

        # 4. Check for traditional measurement units
        units_detected = []
        for unit, unit_info in self.traditional_units.items():
            if unit in verbatim:
                units_detected.append({
                    "unit": unit,
                    "info": unit_info,
                })

        # Determine highest scoring inferred dialect from lexical cues
        max_dialect = max(dialect_scores.items(), key=lambda x: x[1])
        inferred_dialect = max_dialect[0] if max_dialect[1] > 0 else (dialect_hint or "standard")

        # Calculate Diglossia Edit Ratio (how much spoken diverged from written)
        import difflib
        matcher = difflib.SequenceMatcher(None, verbatim, normalized)
        diglossia_distance = round(1.0 - matcher.ratio(), 4)

        return {
            "verbatim": verbatim,
            "normalized": normalized,
            "detected_dialect_cues": list(set(detected_cues)),
            "inferred_dialect": inferred_dialect,
            "terms_mapped": terms_mapped,
            "units_detected": units_detected,
            "diglossia_distance": diglossia_distance,
        }


# Singleton instance
_normalizer = None

def get_diglossia_normalizer() -> DiglossiaNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = DiglossiaNormalizer()
    return _normalizer
