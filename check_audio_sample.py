import wave

# Replace 'your_audio_file.wav' with the actual file path
wav_file_path =r"D:\audio classification\UrbanSound8K\audio\fold1\7061-6-0-0.wav"

# Open the WAV file
with wave.open(wav_file_path, 'rb') as wf:
    # Get sampling rate
    sampling_rate = wf.getframerate()
    # Get bit depth
    bit_depth = wf.getsampwidth() * 8
    # Get number of channels
    num_channels = wf.getnchannels()
    # Get total frames (duration)
    total_frames = wf.getnframes()

print(f"Sampling Rate: {sampling_rate} Hz")
print(f"Bit Depth: {bit_depth} bits")
print(f"Number of Channels: {num_channels}")
print(f"Total Frames (Duration): {total_frames}")
