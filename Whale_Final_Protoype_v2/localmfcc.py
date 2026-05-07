import speech_recognition as sr
import whisper
from sentence_transformers import SentenceTransformer, util
import random
import json
from pydub import AudioSegment
from pydub.playback import play
import warnings
import tkinter as tk
import threading

warnings.filterwarnings("ignore")

# --- LOADING THE BRAIN ---
print("Loading AI Language Models... (This takes a moment)")
whisper_model = whisper.load_model("base.en")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- INSTALLATION DATA ---
human_dictionary = {
    "Machine_Family_0": "Excited",
    "Machine_Family_1": "Breeding song, Communication",
    "Machine_Family_2": "Neutral Conversation",
    "Machine_Family_3": "Related to Food",
    "Machine_Family_4": "Angry, Intimidating",
    "Machine_Family_5": "Happy, Singing",
    "Machine_Family_6": "Whale is treating you like a child",
    "Machine_Family_7": "Sad"
}

print("Loading 753-segment acoustic timeline...")
with open("machine_generated_timeline.json", "r") as f:
    timeline = json.load(f)

print("Loading Audio Files into RAM...")
master_audio = AudioSegment.from_wav("whalefinalsonic.wav")

try:
    special_song = AudioSegment.from_wav("whalesongfeature.wav")
    special_song = special_song.set_frame_rate(master_audio.frame_rate).set_channels(master_audio.channels).set_sample_width(master_audio.sample_width)
except FileNotFoundError:
    print("WARNING: 'whalesongfeature.wav' not found.")
    special_song = None

print("Vectorizing emotional targets...")
emotion_descriptions = list(human_dictionary.values())
emotion_embeddings = embedding_model.encode(emotion_descriptions)

# ---------------------------------------------------------
# 1. UI SETUP (Courier, 14pt, Horizontal, #0090f2 Title)
# ---------------------------------------------------------
root = tk.Tk()
root.title("Installation Interface")

pure_black = "#000000"
pure_white = "#ffffff"
whale_blue = "#0090f2" 
base_font = ("Courier", 14)

root.configure(bg=pure_black)

# Borderless Window Setup (Drag to Screen 2 then press 'F')
root.geometry("1280x720") 
is_fullscreen = False

def toggle_fullscreen(event=None):
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    if is_fullscreen:
        root.overrideredirect(True)
        root.state('zoomed')
    else:
        root.overrideredirect(False)
        root.state('normal')

root.bind("<f>", toggle_fullscreen)
root.bind("<Escape>", lambda e: root.destroy())

main_frame = tk.Frame(root, bg=pure_black)
main_frame.pack(expand=True, fill="both")

row_frame = tk.Frame(main_frame, bg=pure_black)
row_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8)

# UI Elements
title_label = tk.Label(row_frame, text="Whale Communicator", font=base_font, fg=whale_blue, bg=pure_black)
title_label.pack(side="left", expand=True)

status_label = tk.Label(row_frame, text="Speak now", font=base_font, fg=pure_white, bg=pure_black)
status_label.pack(side="right", expand=True)

def set_ui_state(state):
    def _update():
        mapping = {
            'speak': "Speak now",
            'listen': "listening",
            'think': "Thinking...",
            'whale': "whale speaking"
        }
        status_label.configure(text=mapping.get(state, "Speak now"))
    root.after(0, _update)

# ---------------------------------------------------------
# 2. AUDIO & AI LOGIC
# ---------------------------------------------------------
def play_padded_segment(cluster_id):
    family_segments = [seg for seg in timeline if seg["cluster_id"] == cluster_id]
    chosen_segment = random.choice(family_segments)
    
    # RANDOM DURATION LOGIC (Between 4 and 12 seconds)
    start_ms = chosen_segment["start_ms"]
    random_duration = random.randint(4000, 12000)
    end_ms = min(len(master_audio), start_ms + random_duration)
    
    # Console Log for Monitoring
    final_duration = (end_ms - start_ms) / 1000
    print(f"\n[BRAIN]: Matched {cluster_id} ({human_dictionary[cluster_id]})")
    print(f"[BRAIN]: Playing segment: {final_duration:.2f}s")
    
    audio_chunk = master_audio[start_ms : end_ms]
    smoothed_chunk = audio_chunk.fade_in(1000).fade_out(1000)
    
    set_ui_state('whale')
    play(smoothed_chunk)
    set_ui_state('speak')

def listen_and_process():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 150
    recognizer.pause_threshold = 2.5 # Allows for long, natural pauses
    
    with sr.Microphone() as source:
        set_ui_state('listen') # Update UI immediately when mic opens
        try:
            audio = recognizer.listen(source, phrase_time_limit=15)
            set_ui_state('think')
            
            with open("temp_user_input.wav", "wb") as f:
                f.write(audio.get_wav_data())
            
            result = whisper_model.transcribe("temp_user_input.wav", fp16=False)
            return result["text"].strip()
        except Exception as e:
            print(f"Mic Error: {e}")
            return ""

def installation_loop():
    while True:
        set_ui_state('speak')
        user_speech = listen_and_process()
        
        if not user_speech:
            continue

        print(f"\nUSER SAID: {user_speech}")
        
        # Easter Egg
        if "sing" in user_speech.lower() and special_song:
            print("[BRAIN]: Easter Egg Triggered")
            set_ui_state('whale')
            play(special_song)
            set_ui_state('speak')
            continue

        # Semantic Calculation
        user_embed = embedding_model.encode(user_speech)
        scores = util.cos_sim(user_embed, emotion_embeddings)[0]
        best_match_index = scores.argmax().item()
        winning_cluster_id = list(human_dictionary.keys())[best_match_index]
        
        play_padded_segment(winning_cluster_id)

if __name__ == "__main__":
    brain_thread = threading.Thread(target=installation_loop, daemon=True)
    brain_thread.start()
    root.mainloop()