import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr

# Set the paths
input_directory = "D:/input"
output_directory = "D:/input_resampled"

# Specify the target sampling rate
target_sr = 44100

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get a list of all audio files in the input directory
audio_files = [file for file in os.listdir(input_directory) if file.endswith(".wav")]

# Resample, normalize, apply noise reduction, and save each audio file
for audio_file in audio_files:
    input_path = os.path.join(input_directory, audio_file)
    output_path = os.path.join(output_directory, audio_file)

    # Load audio file
    y, sr_original = librosa.load(input_path, sr=None)

    # Resample the audio
    y_resampled = librosa.resample(y, orig_sr=sr_original, target_sr=target_sr)

    # Normalize the audio by dividing by the maximum absolute value
    y_normalized = y_resampled / np.max(np.abs(y_resampled))

    # Apply noise reduction
    y_denoised = nr.reduce_noise(y_normalized, sr=target_sr)

    # Save the resampled, normalized, and noise-reduced audio
    sf.write(output_path, y_denoised, target_sr)

print("Resampling, normalization, and noise reduction complete.")
