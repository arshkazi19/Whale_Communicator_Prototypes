# audiolibrosa_v3.py
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import warnings

warnings.filterwarnings("ignore")

print("Loading 49-minute master audio... (This will take a few minutes)")
y, sr = librosa.load("whalefinalsonic.wav", sr=22050)

# Analyze the audio in 2-second chunks for high resolution
chunk_duration = 2 
chunk_samples = chunk_duration * sr

features = []
timestamps = []

print("Extracting MFCC Timbre fingerprints for every 2-second window...")
for i in range(0, len(y), chunk_samples):
    chunk = y[i : i + chunk_samples]
    
    if len(chunk) < chunk_samples:
        break
        
    # Skip silent ocean noise (Loudness threshold)
    if np.mean(librosa.feature.rms(y=chunk)) < 0.005:
        continue
        
    # Extract MFCCs (Timbre/Texture)
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    avg_mfcc = np.mean(mfccs, axis=1)
    
    features.append(avg_mfcc)
    timestamps.append({"start_ms": int((i / sr) * 1000), "end_ms": int(((i + chunk_samples) / sr) * 1000)})

# ---------------------------------------------------------
# MACHINE LEARNING: K-MEANS CLUSTERING
# ---------------------------------------------------------
print(f"Extracted {len(features)} active acoustic windows. Starting Machine Learning...")

# 1. Normalize the data (Crucial for AI clustering so loud/quiet sounds don't throw it off)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 2. Ask the AI to find 8 distinct "Timbre Families" in the data
# (You can change n_clusters to 5, 10, etc., to get more or fewer groups)
num_clusters = 8
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(features_scaled)

# ---------------------------------------------------------
# CONSOLIDATE AND SAVE THE TIMELINE
# ---------------------------------------------------------
print("Consolidating timeline...")
machine_clusters = []
current_cluster = -1
start_time = 0

# Group back-to-back chunks of the same cluster into long segments
for i, label in enumerate(cluster_labels):
    label = int(label)
    
    if label != current_cluster:
        # If the cluster changes, save the previous one
        if current_cluster != -1:
            machine_clusters.append({
                "cluster_id": f"Machine_Family_{current_cluster}",
                "start_ms": start_time,
                "end_ms": timestamps[i-1]["end_ms"]
            })
        # Start tracking the new cluster
        current_cluster = label
        start_time = timestamps[i]["start_ms"]

# Save the very last segment
if current_cluster != -1:
    machine_clusters.append({
        "cluster_id": f"Machine_Family_{current_cluster}",
        "start_ms": start_time,
        "end_ms": timestamps[-1]["end_ms"]
    })

# Save to JSON
with open("machine_generated_timeline.json", "w") as f:
    json.dump(machine_clusters, f, indent=4)

print(f"\nSuccess! The AI found {num_clusters} unique acoustic families.")
print(f"It built a seamless timeline with {len(machine_clusters)} dynamic audio segments.")
print("Saved to 'machine_generated_timeline.json'")