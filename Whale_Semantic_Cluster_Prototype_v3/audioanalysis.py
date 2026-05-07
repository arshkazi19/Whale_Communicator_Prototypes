from pydub import AudioSegment

master_audio = AudioSegment.from_wav("whalefinalsonic.wav")
print(f"Average Loudness: {master_audio.dBFS:.2f} dBFS")
print(f"Peak Loudness: {master_audio.max_dBFS:.2f} dBFS")