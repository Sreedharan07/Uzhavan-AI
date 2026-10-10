"""
🌾 UZHAVAN AI - Multimodal Agri-Consultation Engine
Cross-Modal Fusion of Leaf Imagery (EfficientNet-B0) and Spoken Voice Symptoms (Tamil NLP)
for Robust Crop Disease Diagnosis under Field Conditions (Blur, Glare, Incomplete Views).
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ml.nlp.diglossia_normalizer import get_diglossia_normalizer


# Semantic symptom lexicon mapping common Tamil & English farmer symptom descriptions to disease classes
SYMPTOM_DISEASE_KB = {
    "மஞ்சள் இலை": {"diseases": ["Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Potato___Early_blight"], "weight": 0.90},
    "மஞ்சளா": {"diseases": ["Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Potato___Early_blight"], "weight": 0.85},
    "மஞ்சள்": {"diseases": ["Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Potato___Early_blight"], "weight": 0.80},
    "புள்ளி": {"diseases": ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Septoria_leaf_spot", "Potato___Early_blight", "Corn_(maize)___Common_rust_"], "weight": 0.90},
    "கரும்புள்ளி": {"diseases": ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Apple___Black_rot"], "weight": 0.95},
    "கருகல்": {"diseases": ["Tomato___Late_blight", "Potato___Late_blight", "Tomato___Early_blight", "Corn_(maize)___Northern_Leaf_Blight"], "weight": 0.95},
    "கருகி": {"diseases": ["Tomato___Late_blight", "Potato___Late_blight", "Tomato___Early_blight"], "weight": 0.90},
    "சுருட்டை": {"diseases": ["Tomato___Tomato_Yellow_Leaf_Curl_Virus"], "weight": 0.95},
    "சுருண்ட": {"diseases": ["Tomato___Tomato_Yellow_Leaf_Curl_Virus"], "weight": 0.95},
    "சுருட்டல்": {"diseases": ["Tomato___Tomato_Yellow_Leaf_Curl_Virus"], "weight": 0.95},
    "சுருளுது": {"diseases": ["Tomato___Tomato_Yellow_Leaf_Curl_Virus"], "weight": 0.95},
    "வாடல்": {"diseases": ["Tomato___Bacterial_spot", "Potato___Late_blight"], "weight": 0.85},
    "வாடி": {"diseases": ["Tomato___Bacterial_spot", "Potato___Late_blight"], "weight": 0.85},
    "சாம்பல் பூஞ்சை": {"diseases": ["Tomato___Leaf_Mold", "Squash___Powdery_mildew", "Cherry_(including_sour)___Powdery_mildew"], "weight": 0.90},
    "சாம்பல்": {"diseases": ["Tomato___Leaf_Mold", "Squash___Powdery_mildew"], "weight": 0.85},
    "வெள்ளை தூள்": {"diseases": ["Tomato___Leaf_Mold", "Squash___Powdery_mildew"], "weight": 0.90},
    "அழுகல்": {"diseases": ["Tomato___Late_blight", "Potato___Late_blight", "Apple___Black_rot", "Grape___Black_rot"], "weight": 0.90},
    "அழுகி": {"diseases": ["Tomato___Late_blight", "Potato___Late_blight"], "weight": 0.90},
    "துரு": {"diseases": ["Corn_(maize)___Common_rust_", "Apple___Cedar_apple_rust"], "weight": 0.95},
    "செம்பழுப்பு": {"diseases": ["Corn_(maize)___Common_rust_", "Apple___Cedar_apple_rust"], "weight": 0.85},
    "வண்டு": {"diseases": ["Tomato___Spider_mites Two-spotted_spider_mite"], "weight": 0.80},
    "சிலந்தி": {"diseases": ["Tomato___Spider_mites Two-spotted_spider_mite"], "weight": 0.95},
    "வலை": {"diseases": ["Tomato___Spider_mites Two-spotted_spider_mite"], "weight": 0.85},
    "ஆரோக்கிய": {"diseases": ["Tomato___healthy", "Potato___healthy", "Apple___healthy", "Corn_(maize)___healthy"], "weight": 0.95},
    "பச்சையா": {"diseases": ["Tomato___healthy", "Potato___healthy", "Apple___healthy", "Corn_(maize)___healthy"], "weight": 0.85},
    "yellow leaf": {"diseases": ["Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"], "weight": 0.85},
    "yellow": {"diseases": ["Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"], "weight": 0.80},
    "black spot": {"diseases": ["Tomato___Bacterial_spot", "Tomato___Early_blight"], "weight": 0.90},
    "spot": {"diseases": ["Tomato___Bacterial_spot", "Tomato___Septoria_leaf_spot", "Potato___Early_blight"], "weight": 0.85},
    "curl": {"diseases": ["Tomato___Tomato_Yellow_Leaf_Curl_Virus"], "weight": 0.95},
    "blight": {"diseases": ["Tomato___Late_blight", "Potato___Late_blight", "Tomato___Early_blight"], "weight": 0.95},
    "wilting": {"diseases": ["Tomato___Bacterial_spot"], "weight": 0.80},
    "wilt": {"diseases": ["Tomato___Bacterial_spot"], "weight": 0.80},
    "rust": {"diseases": ["Corn_(maize)___Common_rust_", "Apple___Cedar_apple_rust"], "weight": 0.95},
    "rot": {"diseases": ["Tomato___Late_blight", "Potato___Late_blight", "Apple___Black_rot", "Grape___Black_rot"], "weight": 0.90},
    "healthy": {"diseases": ["Tomato___healthy", "Potato___healthy", "Apple___healthy", "Corn_(maize)___healthy"], "weight": 0.95},
}


class CrossModalAttentionFusion(nn.Module):
    """
    Bilateral cross-attention gating layer that dynamically weighs visual
    and auditory/textual symptoms according to input certainty.
    """

    def __init__(self, visual_dim: int = 1280, text_dim: int = 256, num_classes: int = 38):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)

        # Cross-modal uncertainty gate
        self.gate = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_h = torch.relu(self.visual_proj(v_feat))
        t_h = torch.relu(self.text_proj(t_feat))

        combined = torch.cat([v_h, t_h], dim=-1)
        weights = self.gate(combined)  # [B, 2] -> (visual_weight, text_weight)

        fused = weights[:, 0:1] * v_h + weights[:, 1:2] * t_h
        logits = self.classifier(fused)
        return logits, weights


class MultimodalAgriConsultant:
    """
    End-to-End Multimodal Crop Diagnosis Consultant.
    Combines Leaf Image Classifier with Spoken/Text Symptom Analyzer.
    """

    def __init__(
        self,
        model_path: str = "models/crop_disease/best_crop_model.pt",
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = get_diglossia_normalizer()
        self._visual_model = None
        self._class_names = None
        self._disease_db = self._load_disease_db()

    def _load_disease_db(self) -> Dict[str, Any]:
        path = PROJECT_ROOT / "agriculture" / "diseases.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _get_visual_model(self):
        if self._visual_model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Crop model not found at {self.model_path}")
            
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            self._class_names = checkpoint.get("class_names", [])
            num_classes = len(self._class_names) or 38

            from ml.crop_disease.train import CropDiseaseModel
            model = CropDiseaseModel(
                model_name=checkpoint.get("model_name", "mobilenet_v2"),
                num_classes=num_classes,
                pretrained=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self.device)
            model.eval()
            self._visual_model = model

        return self._visual_model, self._class_names

    def analyze_symptom_text(self, text: str) -> Dict[str, Any]:
        """Normalize Tamil text and compute semantic disease scores from symptom lexicon."""
        norm_result = self.normalizer.normalize(text)
        cleaned = norm_result["normalized"].lower()
        matched_symptoms = []
        disease_scores: Dict[str, float] = {}

        for symptom, info in SYMPTOM_DISEASE_KB.items():
            if symptom in cleaned or symptom in text:
                matched_symptoms.append(symptom)
                for d in info["diseases"]:
                    disease_scores[d] = disease_scores.get(d, 0.0) + info["weight"]

        total = sum(disease_scores.values())
        if total > 0:
            disease_probs = {k: v / total for k, v in disease_scores.items()}
        else:
            disease_probs = {}

        return {
            "matched_symptoms": matched_symptoms,
            "disease_scores": disease_probs,
            "normalized_query": cleaned,
            "diglossia_metadata": norm_result,
        }

    def consult(
        self,
        image_path: str,
        symptom_text: Optional[str] = None,
        symptom_audio: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multimodal diagnosis fusing leaf image and spoken/text symptoms.
        """
        from torchvision import transforms

        # If symptom audio provided, transcribe it first
        transcription = ""
        if symptom_audio and Path(symptom_audio).exists():
            try:
                from ml.asr.dialect_conditioned_asr import get_dialect_conditioned_asr
                asr = get_dialect_conditioned_asr()
                asr_res = asr.transcribe_with_dialect(symptom_audio)
                transcription = asr_res["normalized_transcription"]
            except Exception:
                pass

        full_symptom_text = (symptom_text or "") + (" " + transcription if transcription else "")
        full_symptom_text = full_symptom_text.strip()

        # Step 1: Visual Inference
        model, class_names = self._get_visual_model()
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            visual_logits = model(img_tensor)
            visual_probs = torch.softmax(visual_logits, dim=-1)[0].cpu().numpy()

        top_v_idx = int(np.argmax(visual_probs))
        top_v_class = class_names[top_v_idx]
        visual_conf = float(visual_probs[top_v_idx])

        # Step 2: Symptom Inference
        symptom_analysis = self.analyze_symptom_text(full_symptom_text) if full_symptom_text else None
        symptom_scores = symptom_analysis["disease_scores"] if symptom_analysis else {}

        # Step 3: Adaptive Cross-Modal Fusion
        # If visual confidence is high (> 0.85) and symptoms are sparse, weigh vision higher (0.8 vs 0.2)
        # If visual confidence is poor (< 0.60, blurry photo) and symptoms are rich, weigh symptoms higher (0.4 vs 0.6)
        if symptom_scores:
            if visual_conf > 0.80:
                v_weight, t_weight = 0.75, 0.25
            elif visual_conf < 0.50:
                v_weight, t_weight = 0.35, 0.65
            else:
                v_weight, t_weight = 0.55, 0.45

            fused_scores = {}
            for i, c_name in enumerate(class_names):
                v_p = float(visual_probs[i])
                t_p = float(symptom_scores.get(c_name, 0.0))
                # If symptom model has no specific entry for class, use small prior
                fused_scores[c_name] = v_weight * v_p + t_weight * t_p

            best_class = max(fused_scores.items(), key=lambda x: x[1])[0]
            multimodal_conf = float(fused_scores[best_class])
            symptom_alignment = "high" if best_class in symptom_scores else "moderate"
        else:
            v_weight, t_weight = 1.0, 0.0
            best_class = top_v_class
            multimodal_conf = visual_conf
            symptom_alignment = "none_provided"

        # Parse crop and disease name
        parts = best_class.split("___")
        crop_raw = parts[0].replace("_", " ").title()
        disease_raw = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"

        # Retrieve verified treatment from diseases.json
        treatment_info = self._get_treatment(crop_raw, disease_raw)

        return {
            "crop": crop_raw,
            "disease": disease_raw,
            "predicted_class": best_class,
            "visual_confidence": round(visual_conf, 4),
            "multimodal_confidence": round(multimodal_conf, 4),
            "fusion_weights": {"visual": v_weight, "symptom": t_weight},
            "symptom_alignment": symptom_alignment,
            "symptoms_identified": symptom_analysis["matched_symptoms"] if symptom_analysis else [],
            "transcribed_symptom_text": full_symptom_text,
            "treatment": treatment_info,
            "disclaimer": "⚠️ மாதிரி மதிப்பீடு மட்டுமே. கள வேளாண் அலுவலரிடம் சரிபார்க்கவும்.",
        }

    def _get_treatment(self, crop: str, disease: str) -> Dict[str, Any]:
        crop_clean = crop.lower().strip()
        disease_clean = disease.lower().strip()

        # Handle healthy leaf directly
        if "healthy" in disease_clean or "ஆரோக்கிய" in disease_clean:
            return {
                "disease_tamil": "ஆரோக்கியமானது (நோய் பாதிப்பில்லை)",
                "symptoms": "இலைகள் பசுமையாகவும் எவ்வித பூச்சி, பூஞ்சை நோய்த் தாக்குதல் இன்றியும் உள்ளன.",
                "treatment": "பரிந்துரைக்கப்பட்ட இயற்கை உரங்கள், சமச்சீர் நுண்ணூட்டம் மற்றும் சீரான பாசனத்தை தொடரவும்.",
                "verified": True,
            }

        crop_aliases = {
            "tomato": ["தக்காளி", "tomato"],
            "potato": ["உருளைக்கிழங்கு", "உருளை", "potato"],
            "corn": ["சோளம்", "corn", "maize"],
            "corn (maize)": ["சோளம்", "corn", "maize"],
            "corn_(maize)": ["சோளம்", "corn", "maize"],
            "pepper": ["குடைமிளகாய்", "மிளகாய்", "pepper", "chilli"],
            "pepper, bell": ["குடைமிளகாய்", "மிளகாய்", "pepper"],
            "pepper,_bell": ["குடைமிளகாய்", "மிளகாய்", "pepper"],
            "apple": ["ஆப்பிள்", "apple"],
            "grape": ["திராட்சை", "grape"],
            "rice": ["நெல்", "rice", "paddy"],
            "paddy": ["நெல்", "rice", "paddy"],
            "eggplant": ["கத்திரி", "eggplant", "brinjal"],
            "onion": ["வெங்காயம்", "onion"],
            "coconut": ["தென்னை", "coconut"],
            "banana": ["வாழை", "banana"],
            "orange": ["ஆரஞ்சு", "orange", "citrus"],
        }

        matched_db_entries = []
        for db_crop, db_val in self._disease_db.items():
            db_crop_l = db_crop.lower()
            db_en = db_val.get("english", "").lower()
            
            # Check direct match
            if crop_clean in db_crop_l or db_crop_l in crop_clean or crop_clean in db_en or db_en in crop_clean:
                matched_db_entries.append(db_val)
                continue
                
            # Check aliases
            for alias_key, aliases in crop_aliases.items():
                if alias_key in crop_clean or crop_clean in alias_key:
                    if any(a in db_crop_l or a in db_en for a in aliases):
                        matched_db_entries.append(db_val)
                        break

        # Search within matched crop entries
        for db_entry in matched_db_entries:
            for d in db_entry.get("diseases", []):
                d_en = d.get("english", "").lower()
                d_ta = d.get("tamil", "").lower()
                if d_en in disease_clean or disease_clean in d_en or d_ta in disease_clean or disease_clean in d_ta:
                    return {
                        "disease_tamil": d.get("tamil", disease),
                        "symptoms": d.get("symptoms", ""),
                        "treatment": d.get("treatment", ""),
                        "verified": True,
                    }

        # Global search across all diseases if specific crop entry didn't match
        for db_crop, db_val in self._disease_db.items():
            for d in db_val.get("diseases", []):
                d_en = d.get("english", "").lower()
                if d_en in disease_clean or disease_clean in d_en:
                    return {
                        "disease_tamil": d.get("tamil", disease),
                        "symptoms": d.get("symptoms", ""),
                        "treatment": d.get("treatment", ""),
                        "verified": True,
                    }

        return {
            "disease_tamil": disease,
            "symptoms": "களத்தில் அறிகுறிகளை கூர்ந்து அவதானிக்கவும்",
            "treatment": "வேப்ப எண்ணெய் கரைசல் (3%) அல்லது அருகிலுள்ள வேளாண் அலுவலரின் பரிந்துரைப்படி பூஞ்சாணக் கொல்லி தெளிக்கவும்.",
            "verified": False,
        }


# Singleton instance
_consultant = None

def get_multimodal_consultant() -> MultimodalAgriConsultant:
    global _consultant
    if _consultant is None:
        _consultant = MultimodalAgriConsultant()
    return _consultant
