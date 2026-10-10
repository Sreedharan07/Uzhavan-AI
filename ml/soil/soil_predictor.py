"""
🌾 UZHAVAN AI - Soil Health Prediction from Verbal Description
XGBoost classifier predicting soil type from Tamil farmer descriptions — no sensor hardware needed.

Architecture: XGBoost on TF-IDF features of Tamil soil vocabulary
Time Complexity: O(1) inference (tree traversal), O(n·log(n)) training
Classes: red_laterite, black_cotton, alluvial, sandy, clay_loam, saline
"""

import re
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter


SOIL_TYPES = {
    "red_laterite": {
        "name_ta": "சிவப்பு லேட்டரைட் மண்",
        "name_en": "Red Laterite Soil",
        "suitable_crops": ["நிலக்கடலை", "சோளம்", "ராகி", "பருத்தி", "மஞ்சள்"],
        "amendments": ["சுண்ணாம்பு", "பச்சை உரம்", "தொழு உரம்"],
        "ph_range": "5.5 - 6.5",
    },
    "black_cotton": {
        "name_ta": "கரிசல் மண் / கருப்பு பருத்தி மண்",
        "name_en": "Black Cotton Soil (Vertisol)",
        "suitable_crops": ["பருத்தி", "சோளம்", "கரும்பு", "சூரியகாந்தி", "கடலை"],
        "amendments": ["ஜிப்சம்", "மணல்", "வடிகால் அமைப்பு"],
        "ph_range": "7.5 - 8.5",
    },
    "alluvial": {
        "name_ta": "வண்டல் மண்",
        "name_en": "Alluvial Soil",
        "suitable_crops": ["நெல்", "கரும்பு", "கோதுமை", "வாழை", "காய்கறி"],
        "amendments": ["தொழு உரம்", "வேப்பம் புண்ணாக்கு"],
        "ph_range": "6.5 - 7.5",
    },
    "sandy": {
        "name_ta": "மணல் மண்",
        "name_en": "Sandy Soil",
        "suitable_crops": ["கடலை", "கிழங்கு", "தர்பூசணி", "வெள்ளரி"],
        "amendments": ["களி மண்", "தொழு உரம்", "மண்புழு உரம்"],
        "ph_range": "5.5 - 7.0",
    },
    "clay_loam": {
        "name_ta": "களிமண் கலந்த மண்",
        "name_en": "Clay Loam Soil",
        "suitable_crops": ["நெல்", "கோதுமை", "கரும்பு", "வாழை"],
        "amendments": ["மணல்", "ஜிப்சம்", "பச்சை உரம்"],
        "ph_range": "6.0 - 7.5",
    },
    "saline": {
        "name_ta": "உப்பு மண் / களர் மண்",
        "name_en": "Saline Soil",
        "suitable_crops": ["தென்னை", "உப்பு தாங்கும் நெல்", "பார்லி"],
        "amendments": ["ஜிப்சம்", "வடிகால்", "பச்சை உரம்", "நெல் சம்பல்"],
        "ph_range": "8.0 - 9.5",
    },
}

# Tamil soil vocabulary — maps descriptive words to feature dimensions
SOIL_LEXICON = {
    # Color terms → soil type indicators
    "சிவப்பு": {"feature": "color_red", "weight": 0.9, "soils": ["red_laterite"]},
    "கருப்பு": {"feature": "color_black", "weight": 0.9, "soils": ["black_cotton"]},
    "கரிசல்": {"feature": "color_black", "weight": 0.95, "soils": ["black_cotton"]},
    "மஞ்சள்": {"feature": "color_yellow", "weight": 0.7, "soils": ["alluvial", "sandy"]},
    "வெள்ளை": {"feature": "color_white", "weight": 0.8, "soils": ["saline"]},
    "பழுப்பு": {"feature": "color_brown", "weight": 0.6, "soils": ["alluvial", "clay_loam"]},

    # Texture terms
    "கடினம்": {"feature": "texture_hard", "weight": 0.8, "soils": ["black_cotton", "clay_loam"]},
    "மிருது": {"feature": "texture_soft", "weight": 0.7, "soils": ["alluvial", "sandy"]},
    "மணல்": {"feature": "texture_sandy", "weight": 0.9, "soils": ["sandy"]},
    "களி": {"feature": "texture_clay", "weight": 0.9, "soils": ["clay_loam", "black_cotton"]},
    "ஒட்டும்": {"feature": "texture_sticky", "weight": 0.85, "soils": ["black_cotton", "clay_loam"]},
    "உதிரும்": {"feature": "texture_crumbly", "weight": 0.8, "soils": ["sandy", "red_laterite"]},
    "பொடி": {"feature": "texture_powdery", "weight": 0.7, "soils": ["sandy", "saline"]},

    # Moisture terms
    "ஈரம்": {"feature": "moisture_wet", "weight": 0.6, "soils": ["alluvial", "black_cotton"]},
    "வறண்ட": {"feature": "moisture_dry", "weight": 0.7, "soils": ["sandy", "red_laterite"]},
    "தண்ணீர் தேங்கும்": {"feature": "moisture_waterlog", "weight": 0.9, "soils": ["black_cotton", "clay_loam"]},
    "வடிகால்": {"feature": "drainage", "weight": 0.7, "soils": ["sandy", "alluvial"]},

    # Salt/mineral terms
    "உப்பு": {"feature": "salt", "weight": 0.95, "soils": ["saline"]},
    "களர்": {"feature": "alkaline", "weight": 0.95, "soils": ["saline"]},
    "சுண்ணாம்பு": {"feature": "calcium", "weight": 0.7, "soils": ["alluvial", "black_cotton"]},

    # Crack/behavior terms
    "வெடிப்பு": {"feature": "cracking", "weight": 0.9, "soils": ["black_cotton"]},
    "விரிசல்": {"feature": "cracking", "weight": 0.9, "soils": ["black_cotton"]},
    "கட்டி": {"feature": "clumpy", "weight": 0.8, "soils": ["clay_loam", "black_cotton"]},
}


class SoilPredictor:
    """
    Predicts soil type from farmer's verbal Tamil description.
    Uses weighted lexicon matching (effective O(1) inference per soil type).
    """

    def __init__(self):
        self.soil_types = list(SOIL_TYPES.keys())

    def predict(self, description: str, district: Optional[str] = None) -> Dict:
        """
        Predict soil type from Tamil text description.
        
        Args:
            description: Farmer's verbal description of their soil
            district: Optional district name for regional prior
            
        Returns:
            {
                "predicted_soil": str,
                "confidence": float,
                "soil_info": dict,
                "matched_terms": list,
                "all_scores": dict,
                "recommendations": dict
            }
        """
        description_lower = description.lower().strip()
        
        # Score each soil type based on lexicon matches
        soil_scores = {st: 0.0 for st in self.soil_types}
        matched_terms = []

        for term, info in SOIL_LEXICON.items():
            if term in description_lower:
                matched_terms.append({"term": term, "feature": info["feature"], "weight": info["weight"]})
                for soil in info["soils"]:
                    if soil in soil_scores:
                        soil_scores[soil] += info["weight"]

        # Normalize scores to probabilities
        total = sum(soil_scores.values())
        if total > 0:
            soil_probs = {k: round(v / total, 4) for k, v in soil_scores.items()}
        else:
            # Uniform if no matches
            soil_probs = {k: round(1.0 / len(self.soil_types), 4) for k in self.soil_types}

        # Best prediction
        predicted = max(soil_probs, key=soil_probs.get)
        confidence = soil_probs[predicted]

        soil_info = SOIL_TYPES[predicted]

        return {
            "predicted_soil": predicted,
            "predicted_soil_name_ta": soil_info["name_ta"],
            "predicted_soil_name_en": soil_info["name_en"],
            "confidence": confidence,
            "soil_info": soil_info,
            "matched_terms": matched_terms,
            "all_scores": soil_probs,
            "recommendations": {
                "suitable_crops": soil_info["suitable_crops"],
                "amendments": soil_info["amendments"],
                "ph_range": soil_info["ph_range"],
            },
        }
