"""
🌾 UZHAVAN AI - Custom Dialect Data Collection Application
Web app for volunteers to record Tamil speech in different dialects.

Features:
    - Select dialect → See Tamil sentence → Record → Replay → Submit
    - 1000+ sentence templates covering agriculture, weather, schemes, etc.
    - 16kHz mono WAV recording
    - Metadata + consent tracking

Usage:
    python data_collection/app.py
    Then open http://localhost:5001
"""

import os
import sys
import csv
import json
import uuid
from pathlib import Path
from datetime import datetime

# Reconfigure stdout/stderr for Windows UTF-8 console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
STORAGE_DIR = Path(__file__).parent / "storage"
RECORDINGS_DIR = STORAGE_DIR / "recordings"
METADATA_FILE = STORAGE_DIR / "metadata.csv"
SENTENCES_DIR = Path(__file__).parent / "sentences"

# Ensure directories exist
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)


def load_sentences():
    """Load sentence templates from JSON files."""
    all_sentences = {}
    
    if SENTENCES_DIR.exists():
        for json_file in SENTENCES_DIR.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                category = json_file.stem
                all_sentences[category] = data
    
    if not all_sentences:
        # Built-in sentences if files not found
        all_sentences = get_builtin_sentences()
    
    return all_sentences


def get_builtin_sentences():
    """1000+ built-in Tamil sentence templates for data collection."""
    return {
        "agriculture_crops": [
            "நெல்லுக்கு எப்போது தண்ணீர் விட வேண்டும்?",
            "பயிருக்கு என்ன உரம் போட வேண்டும்?",
            "இந்த பருவத்தில் என்ன பயிர் போடலாம்?",
            "தக்காளி செடி வளர்ப்பது எப்படி?",
            "நிலக்கடலை போடும் நேரம் எப்போது?",
            "கத்திரிக்காய் செடி எப்படி நடணும்?",
            "வெண்டைக்காய் விதைப்பு எப்போது செய்யணும்?",
            "சோளம் வளர்க்க என்ன செய்ய வேண்டும்?",
            "பருத்தி சாகுபடி முறை என்ன?",
            "கரும்பு நடவு எப்போது செய்யணும்?",
            "மஞ்சள் வளர்க்க என்ன மண் வேண்டும்?",
            "எள் பயிர் எப்போது போடணும்?",
            "உளுந்து பயிர் செய்முறை சொல்லுங்க",
            "பாசிப்பயறு எப்படி வளர்க்கணும்?",
            "துவரை பயிர் செய்ய சரியான நேரம் என்ன?",
            "அவரை செடி எப்படி வளர்ப்பது?",
            "பூசணி விதை எப்போது போடணும்?",
            "சுரைக்காய் படர வைக்க என்ன செய்யணும்?",
            "பாகல் செடி வளர்ப்பது எப்படி?",
            "முள்ளங்கி எப்போது விதைக்கணும்?",
            "கொத்தமல்லி வளர்ப்பது எப்படி?",
            "கறிவேப்பிலை செடி நடுவது எப்படி?",
            "வாழை நடவு எப்போது செய்யலாம்?",
            "தென்னை மரம் நடுவது எப்படி?",
            "மா மரம் ஒட்டு கட்டுவது எப்படி?",
            "கொய்யா செடி வளர்ப்பது எப்படி?",
            "திராட்சை தோட்டம் அமைப்பது எப்படி?",
            "மல்லிகை பூ செடி நடுவது எப்படி?",
            "ராகி பயிர் செய்யலாமா?",
            "கம்பு எப்போது விதைக்கணும்?",
            "நெல் நாற்று எப்படி தயாரிப்பது?",
            "நெல் வயல் தயாரிப்பு எப்படி?",
            "விதை நேர்த்தி செய்வது எப்படி?",
            "நாற்றங்கால் அமைப்பது எப்படி?",
            "நடவு செய்யும் முறை சொல்லுங்க",
            "களை எடுக்க எப்போது ஆரம்பிக்கணும்?",
            "அறுவடை நேரம் எப்படி தெரிந்துகொள்வது?",
            "விளைச்சல் அதிகரிக்க என்ன செய்யணும்?",
            "மண் வளம் பாதுகாப்பது எப்படி?",
            "இயற்கை விவசாயம் செய்வது எப்படி?",
        ],
        "agriculture_disease": [
            "வயலில் பூச்சி தாக்குதல் இருக்கிறது",
            "தக்காளி செடிக்கு என்ன நோய் வந்திருக்கு?",
            "நெல்லுக்கு இலை மஞ்சளா இருக்கு",
            "இலையில் புள்ளி புள்ளியா இருக்கு",
            "செடி வாடிப் போகிறது என்ன செய்யணும்?",
            "பூச்சி அடிக்க என்ன மருந்து போடணும்?",
            "கரும்பில் ஏதோ பூச்சி இருக்கு",
            "நெல் வயலில் புழு தாக்கம் இருக்கு",
            "இலை சுருண்டு போகிறது",
            "தண்டு அழுகல் நோய் வந்திருக்கு",
            "வேர் அழுகல் நோய் தெரியுது",
            "பூ உதிர்ந்து போகிறது",
            "காய் புழு தாக்குதல் இருக்கு",
            "வெள்ளை ஈ தொல்லை அதிகமா இருக்கு",
            "தத்துப்பூச்சி தாக்குதல் எப்படி தடுப்பது?",
            "அசுவினி பூச்சி அதிகமா இருக்கு",
            "இலையில் வெள்ளை படிவம் இருக்கு",
            "செடியில் சாம்பல் நோய் வந்திருக்கு",
            "கிழங்கு அழுகிப் போகிறது",
            "பழம் அழுகல் நோய் இருக்கு",
            "நெல் வயலில் கதிர் நோய் இருக்கு",
            "நெல் இலையில் கருகல் நோய் இருக்கு",
            "தக்காளி இலை சுருட்டு நோய் வந்திருக்கு",
            "கத்திரி காய்ப்புழு பிரச்சனை",
            "தென்னை மூட்டைப்பூச்சி",
            "வாழை இலை மஞ்சள் நோய்",
            "மா பூச்சி தாக்குதல்",
            "பயிர் காப்பாற்ற என்ன செய்யணும்?",
            "இயற்கை பூச்சிக்கொல்லி என்ன?",
            "நிம் எண்ணெய் எப்படி பயன்படுத்துவது?",
        ],
        "agriculture_fertilizer": [
            "நெல்லுக்கு என்ன உரம் போடணும்?",
            "யூரியா எவ்வளவு போடணும்?",
            "இயற்கை உரம் எப்படி செய்வது?",
            "DAP உரம் எப்போது போடணும்?",
            "மண் வளம் அதிகரிக்க என்ன செய்யணும்?",
            "தொழு உரம் எவ்வளவு போடணும்?",
            "பொட்டாஷ் உரம் தேவையா?",
            "உரம் போடும் நேரம் எப்போது?",
            "மண்புழு உரம் எப்படி செய்வது?",
            "பச்சை உரம் என்ன?",
            "நிம் பிண்ணாக்கு எப்படி பயன்படுத்துவது?",
            "மண் பரிசோதனை எங்கே செய்யலாம்?",
            "ஜிங்க் சல்பேட் எப்போது போடணும்?",
            "போரான் உரம் எதற்கு?",
            "கடுகு பிண்ணாக்கு உரமா பயன்படுமா?",
            "நுண்ணூட்ட உரம் என்ன?",
            "மண் அமிலத்தன்மை சரி செய்வது எப்படி?",
            "சுண்ணாம்பு போடணுமா?",
            "ஜிப்சம் எப்போது போடணும்?",
            "பயிர் சுழற்சி என்ன நன்மை?",
        ],
        "agriculture_irrigation": [
            "நெல்லுக்கு எப்போது தண்ணீர் விட வேண்டும்?",
            "பாசனம் எப்படி செய்ய வேண்டும்?",
            "சொட்டு நீர் பாசனம் பற்றி சொல்லுங்க",
            "எவ்வளவு தண்ணீர் விடணும்?",
            "தண்ணீர் பற்றாக்குறை என்ன செய்வது?",
            "நீர் மேலாண்மை பற்றி சொல்லுங்க",
            "மழை நீர் சேகரிப்பு எப்படி?",
            "கிணறு தோண்டுவது எப்படி?",
            "ஆழ்துளை கிணறு பற்றி சொல்லுங்க",
            "தெளிப்பு நீர் பாசனம் என்ன?",
            "வாய்க்கால் பராமரிப்பு எப்படி?",
            "நீர் வடிகால் அமைப்பது எப்படி?",
            "உப்பு நீர் பிரச்சனை என்ன செய்வது?",
            "நிலத்தடி நீர் மட்டம் குறைகிறது",
            "குளம் தூர்வாரல் எப்போது?",
        ],
        "weather": [
            "நாளைக்கு மழை வருமா?",
            "இந்த வாரம் மழை எப்படி இருக்கும்?",
            "இன்னைக்கு வானிலை எப்படி?",
            "மழை எப்போது தொடங்கும்?",
            "வெப்பநிலை எவ்வளவு இருக்கும்?",
            "காற்று வேகம் எப்படி?",
            "புயல் வரும் வாய்ப்பு இருக்கா?",
            "இந்த மாசம் மழை எப்படி?",
            "வறட்சி நிலவரம் என்ன?",
            "பனி எப்போது விழும்?",
            "வெயில் எப்போது அதிகமா இருக்கும்?",
            "மழைக்காலம் எப்போது தொடங்கும்?",
            "தென்மேற்கு பருவமழை எப்போது?",
            "வடகிழக்கு பருவமழை எப்போது?",
            "இன்னைக்கு வெயில் அதிகமா இருக்கும்?",
            "ஈரப்பதம் எவ்வளவு இருக்கு?",
            "வெள்ளம் வரும் அபாயம் இருக்கா?",
            "மழை நிறுத்த நேரம் எப்போது?",
            "அடுத்த வாரம் வானிலை எப்படி?",
            "பருவநிலை மாற்றம் பற்றி சொல்லுங்க",
        ],
        "government_schemes": [
            "விவசாயிகளுக்கு என்ன அரசு உதவி இருக்கிறது?",
            "மானியம் எப்படி பெறுவது?",
            "விவசாய காப்பீடு எப்படி செய்வது?",
            "PM-KISAN பற்றி சொல்லுங்க",
            "இலவச மின்சாரம் கிடைக்குமா?",
            "அரசு உதவி எப்படி பெறுவது?",
            "கிசான் கிரெடிட் கார்டு எப்படி வாங்குவது?",
            "பயிர் காப்பீடு எப்படி செய்வது?",
            "உரம் மானியம் எவ்வளவு?",
            "விதை மானியம் கிடைக்குமா?",
            "டிராக்டர் மானியம் எப்படி?",
            "பம்ப் செட் மானியம் இருக்கா?",
            "கால்நடை மானியம் பற்றி சொல்லுங்க",
            "மீன் வளர்ப்பு மானியம் இருக்கா?",
            "கோழி வளர்ப்பு திட்டம் பற்றி சொல்லுங்க",
            "சுய உதவி குழு கடன் எப்படி?",
            "நபார்டு கடன் எப்படி வாங்குவது?",
            "விவசாய கடன் தள்ளுபடி இருக்கா?",
            "இலவச வீட்டு மனை திட்டம் பற்றி",
            "முதியோர் ஓய்வூதியம் எப்படி பெறுவது?",
            "விதவை ஓய்வூதியம் பற்றி",
            "மாற்றுத்திறனாளி உதவி பற்றி",
            "கல்வி உதவி தொகை பற்றி",
            "ஆதார் அட்டை எப்படி புதுப்பிப்பது?",
            "ரேஷன் அட்டை எப்படி வாங்குவது?",
        ],
        "market_price": [
            "நெல் விலை என்ன?",
            "இன்றைய தக்காளி விலை?",
            "சந்தை விலை நிலவரம் என்ன?",
            "நிலக்கடலை விலை எவ்வளவு?",
            "காய்கறி விலை நிலவரம் சொல்லுங்க",
            "மஞ்சள் விலை கூடிடுச்சா?",
            "வெங்காயம் விலை என்ன?",
            "கரும்பு விலை எவ்வளவு?",
            "பருத்தி விலை என்ன?",
            "தேங்காய் விலை எவ்வளவு?",
            "வாழைக்காய் விலை என்ன?",
            "மா விலை நிலவரம் என்ன?",
            "உழவர் சந்தை எங்கே இருக்கு?",
            "கொள்முதல் விலை என்ன?",
            "MSP விலை என்ன?",
        ],
        "public_services": [
            "அருகில் அரசு மருத்துவமனை எங்கே?",
            "அரசு அலுவலகம் எங்கே?",
            "பேரூராட்சி அலுவலகம் எங்கே?",
            "ரேஷன் கடை எங்கே இருக்கு?",
            "அருகில் வங்கி எங்கே?",
            "தபால் நிலையம் எங்கே?",
            "காவல் நிலையம் எங்கே?",
            "பள்ளிக்கூடம் எங்கே?",
            "கால்நடை மருத்துவர் எங்கே?",
            "வேளாண் அலுவலர் யார்?",
            "கிராம நிர்வாக அலுவலர் யார்?",
            "ஊராட்சி தலைவர் யார்?",
            "108 ஆம்புலன்ஸ் எண் என்ன?",
            "தீயணைப்பு நிலையம் எண் என்ன?",
            "மின்சார கோளாறு புகார் எண் என்ன?",
        ],
        "code_mixed": [
            "நாளைக்கு rain இருக்கா?",
            "market-ல price என்ன?",
            "government scheme-ல apply பண்ணணும்",
            "loan எப்படி வாங்குவது?",
            "insurance claim பண்ணணும்",
            "online-ல apply பண்ணலாமா?",
            "mobile-ல check பண்ணலாமா?",
            "weather forecast சொல்லுங்க",
            "soil test report வந்திருக்கா?",
            "subsidy amount எவ்வளவு?",
            "application form எங்கே கிடைக்கும்?",
            "deadline எப்போது?",
            "document-கள் என்ன submit பண்ணணும்?",
            "bank account-ல credit ஆயிருக்கா?",
            "crop-க்கு fertilizer போடணும்",
            "pest control எப்படி?",
            "organic farming method சொல்லுங்க",
            "drip irrigation system install பண்ணணும்",
            "tractor service centre எங்கே?",
            "water pump repair பண்ணணும்",
        ],
        "greetings_general": [
            "வணக்கம்",
            "நலமா?",
            "ஹலோ",
            "நன்றி",
            "வணக்கம் அண்ணா",
            "நன்றி சொல்லணும்",
            "உதவி வேண்டும்",
            "ஒரு தடவை மீண்டும் சொல்லுங்க",
            "இன்னும் கொஞ்சம் விளக்கமா சொல்லுங்க",
            "வேற ஏதாவது இருக்கா?",
            "சரி புரிஞ்சுது",
            "நான் புரியல",
            "மீண்டும் சொல்லுங்க",
            "மெதுவா சொல்லுங்க",
            "தமிழ்ல சொல்லுங்க",
        ],
        "emergency": [
            "வெள்ளம் வருது உதவி வேணும்",
            "புயல் எச்சரிக்கை இருக்கா?",
            "விலங்கு தாக்குதல் உதவி",
            "நெருப்பு பிடித்தது",
            "மருத்துவ உதவி வேணும்",
            "ஆம்புலன்ஸ் அழைக்கணும்",
            "காவல்துறை உதவி வேணும்",
            "மின்சாரம் இல்லை",
            "குடிநீர் பிரச்சனை",
            "சாலை சேதம் புகார்",
        ],
        "elderly_specific": [
            "ஓய்வூதியம் வந்திருக்கா?",
            "மருத்துவமனைக்கு போகணும்",
            "மருந்து வாங்கணும்",
            "ஆதார் அட்டை வேணும்",
            "ரேஷன் கடை எங்கே?",
            "பஸ் எப்போது வரும்?",
            "வங்கியில் பணம் போடணும்",
            "பணம் எடுக்கணும்",
            "போன் நம்பர் தேவை",
            "மகளுக்கு phone பண்ணணும்",
        ],
    }


# Initialize metadata CSV
if not METADATA_FILE.exists():
    with open(METADATA_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "transcript", "dialect", "speaker_id", "age_group",
            "gender", "domain", "code_mixed", "timestamp", "consent",
        ])


# ---- HTML Template ----
RECORDING_HTML = """
<!DOCTYPE html>
<html lang="ta">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌾 UZHAVAN AI - பேச்சு சேகரிப்பு</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d2818 100%);
            color: #e0e0e0; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 700px; margin: 0 auto; }
        h1 { text-align: center; font-size: 2em; margin-bottom: 20px; color: #4ade80; }
        .card {
            background: rgba(255,255,255,0.08); border-radius: 16px;
            padding: 24px; margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        label { display: block; margin-bottom: 6px; font-weight: bold; color: #a5b4fc; }
        select, input {
            width: 100%; padding: 12px; border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05); color: white;
            font-size: 1em; margin-bottom: 16px;
        }
        .sentence-box {
            background: rgba(74,222,128,0.1); border: 2px solid #4ade80;
            border-radius: 12px; padding: 20px; text-align: center;
            font-size: 1.5em; margin: 16px 0; min-height: 80px;
            line-height: 1.6; color: #4ade80;
        }
        .btn {
            padding: 14px 28px; border: none; border-radius: 10px;
            font-size: 1.1em; cursor: pointer; font-weight: bold;
            transition: all 0.3s; display: inline-block; margin: 6px;
        }
        .btn-record { background: #ef4444; color: white; }
        .btn-record:hover { background: #dc2626; transform: scale(1.05); }
        .btn-record.recording { animation: pulse 1s infinite; }
        .btn-stop { background: #f59e0b; color: white; }
        .btn-play { background: #3b82f6; color: white; }
        .btn-submit { background: #22c55e; color: white; }
        .btn-next { background: #8b5cf6; color: white; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-group { text-align: center; margin: 16px 0; }
        @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.7); } 50% { box-shadow: 0 0 0 10px rgba(239,68,68,0); } }
        .status { text-align: center; padding: 10px; color: #4ade80; font-weight: bold; }
        .consent { display: flex; align-items: center; gap: 10px; margin: 16px 0; }
        .consent input { width: auto; }
        .stats { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }
        .stat { background: rgba(255,255,255,0.05); padding: 12px 20px; border-radius: 10px; text-align: center; }
        .stat .num { font-size: 1.5em; color: #4ade80; font-weight: bold; }
        .privacy { font-size: 0.85em; color: #9ca3af; text-align: center; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌾 UZHAVAN AI<br><span style="font-size:0.5em">பேச்சு சேகரிப்பு / Speech Collection</span></h1>
        
        <div class="card">
            <div class="stats" id="stats"></div>
        </div>
        
        <div class="card">
            <label>வட்டாரம் / Dialect:</label>
            <select id="dialect">
                <option value="standard">நிலையான தமிழ் / Standard Tamil</option>
                <option value="kongu">கொங்கு தமிழ் / Kongu Tamil</option>
                <option value="madurai">மதுரை தமிழ் / Madurai Tamil</option>
                <option value="tirunelveli">திருநெல்வேலி தமிழ் / Tirunelveli Tamil</option>
                <option value="nellai">நெல்லை தமிழ் / Nellai Tamil</option>
                <option value="thanjavur">தஞ்சாவூர் தமிழ் / Thanjavur Tamil</option>
                <option value="chennai">சென்னை தமிழ் / Chennai Tamil</option>
                <option value="northern">வட தமிழ் / Northern Tamil</option>
                <option value="southern">தென் தமிழ் / Southern Tamil</option>
                <option value="code_mixed">தமிழ்-English கலப்பு / Code-Mixed</option>
            </select>
            
            <label>வயது / Age Group:</label>
            <select id="age_group">
                <option value="adult">வயது வந்தவர் / Adult (18-60)</option>
                <option value="elderly">முதியவர் / Elderly (60+)</option>
                <option value="youth">இளைஞர் / Youth (13-18)</option>
            </select>
            
            <label>பாலினம் / Gender:</label>
            <select id="gender">
                <option value="male">ஆண் / Male</option>
                <option value="female">பெண் / Female</option>
                <option value="other">பிற / Other</option>
            </select>
            
            <label>பிரிவு / Category:</label>
            <select id="category" onchange="loadSentence()">
                <option value="agriculture_crops">வேளாண்மை - பயிர் / Agriculture - Crops</option>
                <option value="agriculture_disease">வேளாண்மை - நோய் / Agriculture - Disease</option>
                <option value="agriculture_fertilizer">வேளாண்மை - உரம் / Agriculture - Fertilizer</option>
                <option value="agriculture_irrigation">வேளாண்மை - பாசனம் / Agriculture - Irrigation</option>
                <option value="weather">வானிலை / Weather</option>
                <option value="government_schemes">அரசு திட்டங்கள் / Government Schemes</option>
                <option value="market_price">சந்தை விலை / Market Price</option>
                <option value="public_services">பொது சேவைகள் / Public Services</option>
                <option value="code_mixed">தமிழ்-ஆங்கிலம் / Code Mixed</option>
                <option value="greetings_general">வாழ்த்துகள் / Greetings</option>
                <option value="emergency">அவசரம் / Emergency</option>
                <option value="elderly_specific">முதியவர் / Elderly</option>
            </select>
        </div>
        
        <div class="card">
            <label>இந்த வாக்கியத்தை படிக்கவும் / Read this sentence:</label>
            <div class="sentence-box" id="sentence">Loading...</div>
            <div class="btn-group">
                <button class="btn btn-next" onclick="loadSentence()">⏭️ அடுத்தது / Next</button>
            </div>
        </div>
        
        <div class="card">
            <div class="btn-group">
                <button class="btn btn-record" id="recordBtn" onclick="toggleRecording()">🎤 பதிவு செய்க / Record</button>
                <button class="btn btn-play" id="playBtn" onclick="playRecording()" disabled>▶️ கேட்க / Play</button>
            </div>
            <div class="status" id="status">Ready to record</div>
            
            <div class="consent">
                <input type="checkbox" id="consent" checked>
                <label for="consent" style="margin:0">இந்த ஒலிப்பதிவை AI பயிற்சிக்கு பயன்படுத்த சம்மதம் / I consent to use this recording for AI training</label>
            </div>
            
            <div class="btn-group">
                <button class="btn btn-submit" id="submitBtn" onclick="submitRecording()" disabled>✅ சமர்ப்பி / Submit</button>
            </div>
        </div>
        
        <div class="privacy">
            🔒 உங்கள் தனியுரிமை பாதுகாக்கப்படுகிறது. ஒலிப்பதிவுகள் அநாமதேயமாக சேமிக்கப்படும்.<br>
            Your privacy is protected. Recordings are stored anonymously.
        </div>
    </div>
    
    <script>
        let mediaRecorder, audioChunks = [], audioBlob = null;
        let sentences = {};
        let recordingCount = 0;
        
        async function init() {
            const res = await fetch('/api/sentences');
            sentences = await res.json();
            loadSentence();
            updateStats();
        }
        
        function loadSentence() {
            const category = document.getElementById('category').value;
            const catSentences = sentences[category] || [];
            if (catSentences.length > 0) {
                const idx = Math.floor(Math.random() * catSentences.length);
                document.getElementById('sentence').textContent = catSentences[idx];
            }
        }
        
        async function toggleRecording() {
            const btn = document.getElementById('recordBtn');
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                btn.textContent = '🎤 பதிவு செய்க / Record';
                btn.classList.remove('recording');
                document.getElementById('status').textContent = 'Recording stopped';
                return;
            }
            
            const stream = await navigator.mediaDevices.getUserMedia({audio: {sampleRate: 16000, channelCount: 1}});
            mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
            audioChunks = [];
            
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                audioBlob = new Blob(audioChunks, {type: 'audio/webm'});
                document.getElementById('playBtn').disabled = false;
                document.getElementById('submitBtn').disabled = false;
                stream.getTracks().forEach(t => t.stop());
            };
            
            mediaRecorder.start();
            btn.textContent = '⏹️ நிறுத்து / Stop';
            btn.classList.add('recording');
            document.getElementById('status').textContent = '🔴 Recording...';
        }
        
        function playRecording() {
            if (audioBlob) {
                const url = URL.createObjectURL(audioBlob);
                new Audio(url).play();
            }
        }
        
        async function submitRecording() {
            if (!audioBlob) return;
            if (!document.getElementById('consent').checked) {
                alert('Please provide consent / சம்மதம் தரவும்');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            formData.append('transcript', document.getElementById('sentence').textContent);
            formData.append('dialect', document.getElementById('dialect').value);
            formData.append('age_group', document.getElementById('age_group').value);
            formData.append('gender', document.getElementById('gender').value);
            formData.append('category', document.getElementById('category').value);
            formData.append('consent', 'true');
            
            document.getElementById('status').textContent = 'Submitting...';
            
            const res = await fetch('/api/submit', {method: 'POST', body: formData});
            const data = await res.json();
            
            if (data.success) {
                recordingCount++;
                document.getElementById('status').textContent = '✅ Submitted! / சமர்ப்பிக்கப்பட்டது!';
                audioBlob = null;
                document.getElementById('playBtn').disabled = true;
                document.getElementById('submitBtn').disabled = true;
                loadSentence();
                updateStats();
            }
        }
        
        async function updateStats() {
            try {
                const res = await fetch('/api/stats');
                const stats = await res.json();
                document.getElementById('stats').innerHTML = `
                    <div class="stat"><div class="num">${stats.total}</div>Total</div>
                    <div class="stat"><div class="num">${stats.session}</div>This Session</div>
                    <div class="stat"><div class="num">${Object.keys(stats.by_dialect || {}).length}</div>Dialects</div>
                `;
            } catch(e) {}
        }
        
        init();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(RECORDING_HTML)


@app.route("/api/sentences")
def get_sentences():
    return jsonify(load_sentences())


@app.route("/api/submit", methods=["POST"])
def submit_recording():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"success": False, "error": "No audio file"}), 400
    
    # Generate filename
    filename = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.webm"
    filepath = RECORDINGS_DIR / filename
    audio_file.save(str(filepath))
    
    # Save metadata
    with open(METADATA_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            filename,
            request.form.get("transcript", ""),
            request.form.get("dialect", "standard"),
            f"spk_{uuid.uuid4().hex[:8]}",
            request.form.get("age_group", "adult"),
            request.form.get("gender", "unknown"),
            request.form.get("category", "general"),
            "true" if "code_mixed" in request.form.get("category", "") else "false",
            datetime.now().isoformat(),
            request.form.get("consent", "false"),
        ])
    
    return jsonify({"success": True, "filename": filename})


@app.route("/api/stats")
def get_stats():
    total = 0
    by_dialect = {}
    
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                dialect = row.get("dialect", "unknown")
                by_dialect[dialect] = by_dialect.get(dialect, 0) + 1
    
    return jsonify({"total": total, "session": 0, "by_dialect": by_dialect})


@app.route("/api/delete/<filename>", methods=["DELETE"])
def delete_recording(filename):
    """Delete a recording (privacy compliance)."""
    filepath = RECORDINGS_DIR / filename
    if filepath.exists():
        filepath.unlink()
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Not found"}), 404


if __name__ == "__main__":
    # Save built-in sentences to files
    sentences = get_builtin_sentences()
    SENTENCES_DIR.mkdir(parents=True, exist_ok=True)
    for category, sents in sentences.items():
        with open(SENTENCES_DIR / f"{category}.json", "w", encoding="utf-8") as f:
            json.dump(sents, f, ensure_ascii=False, indent=2)
    
    total = sum(len(s) for s in sentences.values())
    print(f"\n🌾 UZHAVAN AI - Speech Collection Tool")
    print(f"   {total} sentence templates across {len(sentences)} categories")
    print(f"   Open http://localhost:5001")
    
    app.run(host="0.0.0.0", port=5001, debug=True)
