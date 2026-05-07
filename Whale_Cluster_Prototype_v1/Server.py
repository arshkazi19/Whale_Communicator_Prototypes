"""
Whale Decoder — Flask Backend
------------------------------
Flow:
  1. Receive voice recording from browser
  2. Transcribe speech to text (Whisper local model)
  3. Classify text semantically into a category
  4. Return the category + a random 5-10s clip from a whale file in that category

Folder structure:
  whale_library/
    place_location/
    greetings/
    excitement/
    relationships/

Run:
  python server.py

Requirements:
  pip install flask flask-cors sentence-transformers torch openai-whisper soundfile numpy
  ffmpeg must be installed and on your PATH
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import whisper
import torch
import soundfile as sf
import numpy as np
import io
import os
import tempfile
import random

app = Flask(__name__, static_folder=".")
CORS(app)

WHALE_DIR = os.path.join(os.path.dirname(__file__), "whale_library")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

CATEGORIES = {
    "place_location": [
        "This is a place or location.",
        "I am talking about a city, country, or address.",
        "Directions, maps, geography, and physical locations.",
        "Where something is located or situated.",
        "Landmarks, buildings, neighborhoods, regions.",
    ],
    "greetings": [
        "Hello, hi, hey – a greeting or salutation.",
        "Saying goodbye, farewell, or see you later.",
        "A welcoming phrase or pleasantry.",
        "How are you? Nice to meet you.",
        "Good morning, good evening, or a polite opener.",
    ],
    "excitement": [
        "Wow, amazing, incredible – expressing excitement.",
        "Something thrilling, surprising, or astonishing.",
        "I can't believe it! This is awesome!",
        "Enthusiastic reaction to great news or an event.",
        "Feeling pumped up, energized, or overjoyed.",
    ],
    "relationships": [
        "Talking about friends, family, or romantic partners.",
        "Love, marriage, friendship, and human connections.",
        "My mother, father, sibling, or significant other.",
        "Dating, breakups, trust, and emotional bonds.",
        "The relationship between people and how they interact.",
    ],
}

# create category folders
for cat in CATEGORIES:
    os.makedirs(os.path.join(WHALE_DIR, cat), exist_ok=True)

# load models at startup
print("\nLoading Whisper (speech-to-text)...")
whisper_model = whisper.load_model("base")
print("Whisper ready.")

print("Loading sentence-transformer (text classifier)...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

category_embeddings = {}
for category, phrases in CATEGORIES.items():
    embeddings = st_model.encode(phrases, convert_to_tensor=True)
    category_embeddings[category] = embeddings.mean(dim=0)

print("Classifier ready.\n")


# ── helpers ────────────────────────────────────────────────────────────────

def classify_text(text):
    input_emb = st_model.encode(text, convert_to_tensor=True)
    scores = {}
    for cat, centroid in category_embeddings.items():
        scores[cat] = round(util.cos_sim(input_emb, centroid).item(), 4)
    best = max(scores, key=lambda c: scores[c])
    return best, scores


def pick_whale_for_category(category):
    folder = os.path.join(WHALE_DIR, category)
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS]
    if not files:
        return None
    return random.choice(files)


def extract_random_clip(filepath, min_dur=5.0, max_dur=10.0):
    """Read an audio file and return a random 5-10s clip as WAV bytes in memory."""
    data, sr = sf.read(filepath, always_2d=False)
    total_samples = len(data)
    clip_min = int(min_dur * sr)
    clip_max = int(max_dur * sr)

    if total_samples <= clip_min:
        clip = data
    else:
        clip_len = random.randint(
            min(clip_min, total_samples),
            min(clip_max, total_samples)
        )
        start = random.randint(0, total_samples - clip_len)
        clip = data[start:start + clip_len]

    buf = io.BytesIO()
    sf.write(buf, clip, sr, format="WAV")
    buf.seek(0)
    return buf


# ── routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/library", methods=["GET"])
def get_library():
    result = {}
    for cat in CATEGORIES:
        folder = os.path.join(WHALE_DIR, cat)
        result[cat] = sorted([
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ])
    return jsonify(result)


@app.route("/api/library/upload/<category>", methods=["POST"])
def upload_whale(category):
    if category not in CATEGORIES:
        return jsonify({"error": "Unknown category"}), 400
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    folder = os.path.join(WHALE_DIR, category)
    saved = []
    for f in request.files.getlist("files"):
        if os.path.splitext(f.filename)[1].lower() in AUDIO_EXTENSIONS:
            f.save(os.path.join(folder, f.filename))
            saved.append(f.filename)
    return jsonify({"saved": saved, "category": category})


@app.route("/api/library/delete/<category>/<filename>", methods=["DELETE"])
def delete_whale(category, filename):
    path = os.path.join(WHALE_DIR, category, filename)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"deleted": filename})
    return jsonify({"error": "File not found"}), 404


@app.route("/api/translate", methods=["POST"])
def translate():
    if "audio" not in request.files:
        return jsonify({"error": "No audio provided"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"].strip()

        if not transcript:
            return jsonify({"error": "Could not transcribe — try speaking more clearly."}), 400

        category, scores = classify_text(transcript)
        whale_file = pick_whale_for_category(category)

        return jsonify({
            "transcript": transcript,
            "category": category,
            "scores": scores,
            "whale_file": whale_file,
            "clip_url": f"/api/clip/{category}/{whale_file}" if whale_file else None,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/api/clip/<category>/<filename>")
def clip(category, filename):
    """Stream a fresh random 5-10s clip every time it's called."""
    filepath = os.path.join(WHALE_DIR, category, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    try:
        buf = extract_random_clip(filepath)
        return send_file(buf, mimetype="audio/wav", as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Whale Decoder running at http://localhost:5000\n")
    app.run(debug=True, port=5000)