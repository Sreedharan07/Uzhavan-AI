"""
🌾 UZHAVAN AI - Intent Classifier Training
Fine-tunes multilingual BERT (MuRIL) for Tamil agricultural intent classification.

14 Intents: crop_advice, crop_disease, fertilizer, irrigation, weather,
            market_price, government_scheme, subsidy, insurance,
            public_service, emergency, general_question, greeting, unknown

Usage:
    python ml/intent/train.py --config configs/intent.yaml
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# Training data: Tamil sentences for each intent
INTENT_TRAINING_DATA = {
    "crop_advice": [
        "நெல்லுக்கு என்ன செய்யணும்?",
        "இந்த பருவத்தில் என்ன பயிர் போடலாம்?",
        "தக்காளி செடி வளர்ப்பது எப்படி?",
        "கரும்பு பயிர் செய்முறை என்ன?",
        "மஞ்சள் வளர்க்க என்ன மண் வேண்டும்?",
        "நிலக்கடலை போடும் நேரம் எப்போது?",
        "கத்திரி செடி எப்படி நடணும்?",
        "வெண்டைக்காய் விதைப்பு எப்போது?",
        "சோளம் வளர்க்க என்ன செய்ய வேண்டும்?",
        "பருத்தி சாகுபடி முறை சொல்லுங்க",
        "எள் பயிர் செய்யலாமா?",
        "அவரை செடி எப்படி வளர்ப்பது?",
        "what crop to plant this season",
        "how to grow paddy",
        "best time for planting tomato",
    ],
    "crop_disease": [
        "நெல்லுக்கு இலை மஞ்சளா இருக்கு",
        "வயலில் பூச்சி தாக்குதல் இருக்கிறது",
        "தக்காளி செடிக்கு என்ன நோய்?",
        "இலையில் புள்ளி புள்ளியா இருக்கு",
        "செடி வாடிப் போகிறது",
        "பூச்சி அடிக்க என்ன மருந்து?",
        "கரும்பில் ஏதோ பூச்சி இருக்கு",
        "நெல் வயலில் புழு தாக்கம்",
        "இலை சுருண்டு போகிறது",
        "தண்டு அழுகல் நோய் வந்திருக்கு",
        "leaf is turning yellow",
        "pest attack on my crop",
        "what disease is this on plant",
        "worm in paddy field",
    ],
    "fertilizer": [
        "நெல்லுக்கு என்ன உரம் போடணும்?",
        "பயிருக்கு என்ன உரம் போட வேண்டும்?",
        "யூரியா எவ்வளவு போடணும்?",
        "இயற்கை உரம் எப்படி செய்வது?",
        "DAP உரம் எப்போது போடணும்?",
        "மண் வளம் அதிகரிக்க என்ன செய்யணும்?",
        "தொழு உரம் எவ்வளவு போடணும்?",
        "பொட்டாஷ் உரம் தேவையா?",
        "உரம் போடும் நேரம் எப்போது?",
        "which fertilizer for paddy",
        "organic manure preparation",
        "how much urea to use",
    ],
    "irrigation": [
        "நெல்லுக்கு எப்போது தண்ணீர் விட வேண்டும்?",
        "பாசனம் எப்படி செய்ய வேண்டும்?",
        "சொட்டு நீர் பாசனம் பற்றி சொல்லுங்க",
        "எவ்வளவு தண்ணீர் விடணும்?",
        "தண்ணீர் பற்றாக்குறை என்ன செய்வது?",
        "நீர் மேலாண்மை பற்றி சொல்லுங்க",
        "மழை இல்லை தண்ணீர் பிரச்சனை",
        "when to irrigate paddy",
        "drip irrigation method",
        "water management tips",
    ],
    "weather": [
        "நாளைக்கு மழை வருமா?",
        "இந்த வாரம் மழை எப்படி இருக்கும்?",
        "இன்னைக்கு வானிலை எப்படி?",
        "மழை எப்போது தொடங்கும்?",
        "வெப்பநிலை எவ்வளவு?",
        "காற்று வேகம் எப்படி?",
        "tomorrow rain vara?",
        "will it rain this week",
        "weather forecast today",
        "temperature innaikku",
    ],
    "market_price": [
        "நெல் விலை என்ன?",
        "இன்றைய தக்காளி விலை?",
        "சந்தை விலை என்ன?",
        "நிலக்கடலை விலை எவ்வளவு?",
        "காய்கறி விலை நிலவரம்",
        "மஞ்சள் விலை கூடிடுச்சா?",
        "paddy price today",
        "what is tomato rate",
        "market price for groundnut",
    ],
    "government_scheme": [
        "விவசாயிகளுக்கு என்ன அரசு உதவி இருக்கிறது?",
        "அரசு திட்டம் என்ன இருக்கு?",
        "PM-KISAN பற்றி சொல்லுங்க",
        "இலவச மின்சாரம் கிடைக்குமா?",
        "அரசு உதவி எப்படி பெறுவது?",
        "government scheme for farmers",
        "what schemes available",
        "free electricity scheme",
    ],
    "subsidy": [
        "மானியம் எப்படி பெறுவது?",
        "விவசாய மானியம் இருக்கா?",
        "உரம் மானியம் எவ்வளவு?",
        "விதை மானியம் கிடைக்குமா?",
        "டிராக்டர் மானியம் எப்படி?",
        "how to get subsidy",
        "fertilizer subsidy amount",
        "tractor subsidy apply",
    ],
    "insurance": [
        "விவசாய காப்பீடு எப்படி செய்வது?",
        "பயிர் காப்பீடு பற்றி சொல்லுங்க",
        "PMFBY என்ன?",
        "காப்பீட்டு தொகை எவ்வளவு?",
        "crop insurance how to apply",
        "insurance claim process",
    ],
    "public_service": [
        "அருகில் அரசு மருத்துவமனை எங்கே?",
        "அரசு அலுவலகம் எங்கே?",
        "பேரூராட்சி அலுவலகம் எங்கே?",
        "ரேஷன் கடை எங்கே?",
        "nearest government hospital",
        "where is ration shop",
        "panchayat office location",
    ],
    "emergency": [
        "வெள்ளம் வருது உதவி வேணும்",
        "புயல் எச்சரிக்கை",
        "விலங்கு தாக்குதல் உதவி",
        "நெருப்பு பிடித்தது",
        "flood emergency help",
        "storm warning",
        "fire emergency",
    ],
    "general_question": [
        "இது எப்படி வேலை செய்கிறது?",
        "இன்னும் கொஞ்சம் சொல்லுங்க",
        "வேற ஏதாவது இருக்கா?",
        "ஒரு தடவை மீண்டும் சொல்லுங்க",
        "tell me more",
        "how does this work",
        "anything else",
    ],
    "greeting": [
        "வணக்கம்",
        "நலமா?",
        "ஹலோ",
        "நன்றி",
        "வணக்கம் அண்ணா",
        "hello",
        "hi",
        "thanks",
        "nandri",
    ],
    "unknown": [
        "hhkhkh",
        "...",
        "xyzabc",
        "asdfgh",
        "123456",
    ],
}


class IntentDataset(Dataset):
    """Dataset for intent classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class IntentClassifier(nn.Module):
    """BERT-based intent classifier."""
    
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.3):
        super().__init__()
        from transformers import AutoModel
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(pooled)


def prepare_intent_data(
    intents: List[str],
    data: Dict = None,
    custom_train_path: str = None,
) -> tuple:
    """Prepare training data from built-in data + optional custom CSV."""
    data = data or INTENT_TRAINING_DATA
    
    all_texts, all_labels = [], []
    intent_to_idx = {intent: i for i, intent in enumerate(intents)}
    
    # Built-in data
    for intent, sentences in data.items():
        if intent in intent_to_idx:
            for sentence in sentences:
                all_texts.append(sentence)
                all_labels.append(intent_to_idx[intent])
    
    # Custom CSV data
    if custom_train_path and Path(custom_train_path).exists():
        with open(custom_train_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                intent = row.get("intent", "unknown")
                text = row.get("text", "").strip()
                if intent in intent_to_idx and text:
                    all_texts.append(text)
                    all_labels.append(intent_to_idx[intent])
    
    # Shuffle
    indices = list(range(len(all_texts)))
    np.random.seed(42)
    np.random.shuffle(indices)
    all_texts = [all_texts[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split 80/10/10
    n = len(all_texts)
    n_train = int(n * 0.8)
    n_val = int(n * 0.9)
    
    return (
        (all_texts[:n_train], all_labels[:n_train]),
        (all_texts[n_train:n_val], all_labels[n_train:n_val]),
        (all_texts[n_val:], all_labels[n_val:]),
    )


def train_intent_classifier(config: Dict):
    """Train intent classifier."""
    from transformers import AutoTokenizer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["name"]
    intents = config["intents"]
    train_cfg = config["training"]
    
    print("\n" + "=" * 60)
    print("🌾 UZHAVAN AI - Intent Classifier Training")
    print("=" * 60)
    print(f"   Model: {model_name}")
    print(f"   Intents: {len(intents)}")
    print(f"   Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = IntentClassifier(model_name, len(intents), config["model"].get("dropout", 0.3)).to(device)
    
    # Data
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = prepare_intent_data(
        intents, custom_train_path=config.get("data", {}).get("train_path")
    )
    
    print(f"   Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, config["model"].get("max_length", 128))
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, config["model"].get("max_length", 128))
    
    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 2e-5))
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(train_cfg.get("log_dir", "results/intent/tensorboard"))
    best_f1 = 0
    patience_counter = 0
    best_model_path = Path(train_cfg.get("best_model_path", "models/intent/best_intent_model.pt"))
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(train_cfg.get("epochs", 20)):
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validate
        model.eval()
        all_preds, all_labels_val = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_val.extend(batch["label"].numpy())
        
        f1 = f1_score(all_labels_val, all_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_labels_val, all_preds)
        avg_loss = total_loss / max(1, len(train_loader))
        
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("val/f1_macro", f1, epoch)
        writer.add_scalar("val/accuracy", acc, epoch)
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f} | F1={f1:.4f} | Acc={acc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "intents": intents,
                "config": config["model"],
            }, str(best_model_path))
            print(f"  🏆 Best model saved! F1={f1:.4f}")
        else:
            patience_counter += 1
            if train_cfg.get("early_stopping", True) and patience_counter >= train_cfg.get("early_stopping_patience", 5):
                print(f"  ⏹️ Early stopping")
                break
    
    writer.close()
    print(f"\n✅ Intent training complete. Best F1: {best_f1:.4f}")


def classify_intent(
    text: str,
    model_path: str = "models/intent/best_intent_model.pt",
    device: Optional[str] = None,
) -> Dict:
    """Classify intent of Tamil text."""
    from transformers import AutoTokenizer
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(model_path, map_location=device)
    intents = checkpoint["intents"]
    model_name = checkpoint["model_name"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = IntentClassifier(model_name, len(intents), checkpoint["config"].get("dropout", 0.3))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    encoding = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)[0]
    
    pred_idx = probs.argmax().item()
    
    return {
        "intent": intents[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "all_scores": {intents[i]: round(probs[i].item(), 4) for i in range(len(intents))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/intent.yaml")
    parser.add_argument("--text", default=None, help="Classify intent of text")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    if args.text:
        result = classify_intent(args.text)
        print(f"\n💡 Intent: {result['intent']}")
        print(f"   Confidence: {result['confidence']:.2%}")
    else:
        config_path = PROJECT_ROOT / args.config
        with open(config_path) as f:
            config = yaml.safe_load(f)
        if args.epochs:
            config["training"]["epochs"] = args.epochs
        train_intent_classifier(config)


if __name__ == "__main__":
    main()
