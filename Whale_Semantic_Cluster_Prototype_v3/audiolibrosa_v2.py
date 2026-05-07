import librosa
import numpy as np
from pydub import AudioSegment
import warnings

warnings.filterwarnings("ignore")

print("Loading 49-minute master audio... (This takes a moment)")
# Load with librosa for mathematical analysis
y, sr = librosa.load("whalefinalsonic.wav", sr=22050)

# The emotional timeline you already mapped
whale_clusters = [
    {"description": "neutral cute speech, very happy", "start_ms": 0, "end_ms": 125000},
    {"description": "deep groans, pride", "start_ms": 125000, "end_ms": 330000},
    {"description": "neutral conversational, searching food", "start_ms": 330000, "end_ms": 420000},
    {"description": "echo location, calling others", "start_ms": 420000, "end_ms": 938000},
    {"description": "relationships, conversational", "start_ms": 938000, "end_ms": 1140000},
    {"description": "excited and shouting", "start_ms": 1200000, "end_ms": 1800000},
    {"description": "angry and aggressive", "start_ms": 1800000, "end_ms": 1920000},
    {"description": "conversational and happy", "start_ms": 1920000, "end_ms": 2280000},
    {"description": "peaceful, calm, and resting", "start_ms": 2280000, "end_ms": 2939000}
]

print("\n--- EMOTIONAL ACOUSTIC FINGERPRINTS ---")

for cluster in whale_clusters:
    # Convert milliseconds to audio samples
    start_sample = int((cluster["start_ms"] / 1000) * sr)
    end_sample = int((cluster["end_ms"] / 1000) * sr)
    
    # Isolate the specific emotional audio block
    audio_segment = y[start_sample:end_sample]
    
    if len(audio_segment) == 0:
        continue

    # 1. Calculate Average Frequency (Spectral Centroid)
    centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
    avg_freq = np.mean(centroid)
    
    # 2. Calculate Emotion Texture (MFCCs)
    # We grab the first 13 coefficients, which represent the shape of the vocal tract
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    avg_mfcc = np.mean(mfccs, axis=1)
    
    # Calculate the overall "energy" or aggression of the texture
    texture_energy = np.var(avg_mfcc)
    
    print(f"\nEmotion: {cluster['description'].upper()}")
    print(f" -> Dominant Frequency: {avg_freq:.2f} Hz")
    print(f" -> Texture Complexity (Timbre): {texture_energy:.2f}")