import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import wave
import pylab
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import librosa

INPUT_DIR = 'D:\\input\\audio'
OUTPUT_DIR = 'D:\\output_logmel'

def get_mel_spectrogram(wav_file):
    try:
        y, sr = librosa.load(wav_file, sr=None)  # Load the audio file
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)  # Calculate the mel spectrogram
        return mel_spectrogram
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-logMel_spec')):
    os.mkdir(os.path.join(OUTPUT_DIR, 'audio-logMel_spec'))

for filename in os.listdir(INPUT_DIR):
    if "wav" in filename:
        file_path = os.path.join(INPUT_DIR, filename)
        file_stem = Path(file_path).stem
        target_dir = f'class_{file_stem.split("-",1)[0]}'
        dist_dir = os.path.join(os.path.join(OUTPUT_DIR, 'audio-logMel_spec'), target_dir)
        file_dist_path = os.path.join(dist_dir, file_stem)
        if not os.path.exists(file_dist_path + '.png'):
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)
            file_stem = Path(file_path).stem
            mel_spectrogram = get_mel_spectrogram(file_path)  # Get the mel spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.savefig(f'{file_dist_path}.png')
            plt.close()

print("done extracting")