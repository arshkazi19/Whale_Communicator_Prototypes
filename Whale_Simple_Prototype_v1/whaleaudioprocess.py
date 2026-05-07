import librosa
import soundfile as sf
import numpy as np

def isolate_whale_sounds(input_file, output_file, top_db=20):
    """
    Trims silence and isolates high-energy segments from a recording.
    top_db: The threshold (in decibels) below reference to consider as silence.
    Lower top_db = stricter (only the loudest sounds kept).
    """
    print(f"Loading {input_file}...")
    y, sr = librosa.load(input_file, sr=None)

    # 1. Trim leading and trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

    # 2. Split audio into non-silent intervals
    # This identifies exactly where the "action" is
    intervals = librosa.effects.split(y, top_db=top_db)
    
    # 3. Concatenate non-silent parts
    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])

    print(f"Original duration: {librosa.get_duration(y=y, sr=sr):.2f}s")
    print(f"Processed duration: {librosa.get_duration(y=non_silent_audio, sr=sr):.2f}s")

    # 4. Save the "Concentrated Whale" file
    sf.write(output_file, non_silent_audio, sr)
    print(f"Saved to {output_file}")

# Usage
isolate_whale_sounds('WHALE49min.wav', 'clean_whales.wav', top_db=25)