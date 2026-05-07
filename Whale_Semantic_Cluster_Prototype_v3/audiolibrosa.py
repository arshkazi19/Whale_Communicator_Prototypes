# audiolibrosa.py
import librosa
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")

print("Loading 49-minute audio into analyzer (This will take a few minutes)...")
y, sr = librosa.load("whalefinalsonic.wav", sr=22050)

chunk_duration_sec = 5
chunk_samples = chunk_duration_sec * sr
frequency_database = []

print("Analyzing spectral frequencies...")
for i in range(0, len(y), chunk_samples):
    chunk = y[i : i + chunk_samples]
    
    if len(chunk) < chunk_samples: 
        break
        
    centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)
    average_freq = np.mean(centroid)
    
    frequency_database.append({
        "start_ms": int((i / sr) * 1000),
        "freq_hz": float(average_freq)
    })

with open("whale_frequency_map.json", "w") as f:
    json.dump(frequency_database, f)

print(f"Success! Mapped {len(frequency_database)} distinct frequency windows.")