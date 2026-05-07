from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

print("Loading master audio file...")
master_audio = AudioSegment.from_wav("whalefinalsonic.wav")

print("Analyzing and splitting audio...")
chunks = split_on_silence(
    master_audio,
    # CHANGE 1: Increase to 2000ms (2 seconds). 
    # The whale must be completely silent for 2 full seconds before it makes a cut.
    # This prevents a single, stuttering whale call from being chopped into 10 pieces.
    min_silence_len=2000, 
    
    # Keeping the threshold we figured out
    silence_thresh=-22, 
    
    keep_silence=200 
)

output_folder = "whale_slices"
os.makedirs(output_folder, exist_ok=True)

# CHANGE 2: Filter out the garbage. 
# We only want to export slices that are longer than 2 seconds. 
# This throws away all the useless 0.1-second pops and clicks.
minimum_duration_ms = 2000 
saved_count = 0

print(f"Initially found {len(chunks)} raw cuts. Filtering for quality...")

for i, chunk in enumerate(chunks):
    if len(chunk) >= minimum_duration_ms:
        chunk.export(f"{output_folder}/whale_call_{saved_count}.wav", format="wav")
        print(f"Exported chunk {saved_count} - Duration: {len(chunk)/1000.0} seconds")
        saved_count += 1

print(f"\nSuccess! You now have {saved_count} highly usable audio slices.")