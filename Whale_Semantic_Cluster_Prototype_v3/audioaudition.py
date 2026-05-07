# audioaudition.py
import json
import random
from pydub import AudioSegment
from pydub.playback import play
import warnings

warnings.filterwarnings("ignore")

print("Loading Machine Timeline...")
with open("machine_generated_timeline.json", "r") as f:
    timeline = json.load(f)

print("Loading 49-minute master audio... (This takes a moment)")
master_audio = AudioSegment.from_wav("whalefinalsonic.wav")

# Group all the segments into their respective families
clusters = {}
for segment in timeline:
    cid = segment["cluster_id"]
    if cid not in clusters:
        clusters[cid] = []
    clusters[cid].append(segment)

print(f"\n--- AUDITION READY ---")
print(f"Found {len(clusters)} unique acoustic families.")

while True:
    print("\n------------------------------------------------")
    user_input = input("Enter a Family Number (0 to 7) to listen, or 'q' to quit: ")
    
    if user_input.lower() == 'q':
        break
        
    target_cluster = f"Machine_Family_{user_input}"
    
    if target_cluster not in clusters:
        print("Invalid number. Try again.")
        continue
        
    family_segments = clusters[target_cluster]
    print(f"\n{target_cluster} has {len(family_segments)} total audio segments.")
    print("Playing 3 random samples so you can identify the emotion...")
    
    # Pick up to 3 random segments from this family to play
    sample_size = min(3, len(family_segments))
    samples = random.sample(family_segments, sample_size)
    
    for i, sample in enumerate(samples):
        start = sample["start_ms"]
        end = sample["end_ms"]
        dur = (end - start) / 1000.0
        
        print(f"  Playing Sample {i+1} ({dur:.1f} seconds)...")
        
        # Grab the audio, add a tiny fade to prevent speaker pops, and play
        audio_chunk = master_audio[start:end]
        play(audio_chunk.fade_in(50).fade_out(50))
        
    print("\n--> Write down your notes for this family!")