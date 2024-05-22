import librosa
import soundfile as sf

# Load audio clips
audio_dcase, sr_dcase = librosa.load(r"D:\input\audio\airport-barcelona-0-2-c.wav", sr=None)
audio_urbansound, sr_urbansound = librosa.load(r"D:\audio classification\UrbanSound8K\audio\fold1\7061-6-0-0.wav", sr=None)

# Ensure the same length (trim or pad if necessary)
min_len = min(len(audio_dcase), len(audio_urbansound))
audio_dcase = audio_dcase[:min_len]
audio_urbansound = audio_urbansound[:min_len]

# Mix the audio clips
mixed_audio = audio_dcase + audio_urbansound

# Save the mixed audio
sf.write('path_to_mixed_clip.wav', mixed_audio, sr_dcase)