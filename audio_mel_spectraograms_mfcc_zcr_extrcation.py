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

INPUT_DIR = 'D:\\input_resampled'
OUTPUT_DIR = 'D:\\output_resampled'
OUTPUT_DIR_ZCR = 'D:\\output_resampled\\zcr_values'
OUTPUT_DIR_MFCC='D:\\output_resampled\\mfcc_values'
OUTPUT_DIR_SPECTRAL_COEF='D:\\output_resampled\\spec_centroid'
OUTPUT_DIR_SPECTRAL_BW='D:\\output_resampled\\spec_bw'

def get_mel_spectrogram(wav_file):
    try:
        y, sr = librosa.load(wav_file, sr=None)  # Load the audio file
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)  # Calculate the mel spectrogram
        return mel_spectrogram
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_zcr_and_save(audio_file, output_dir):
    try:
        wav = wave.open(audio_file, 'r')
        frames = wav.readframes(-1)
        sound_info = np.frombuffer(frames, dtype=np.int16)
        frame_rate = wav.getframerate()
        wav.close()
        
        # Calculate ZCR
        zcr = np.mean(np.abs(np.diff(np.sign(sound_info))) / 2.0)
        
        # Extract the class name from the audio file name
        file_name = os.path.basename(audio_file)
        class_name = file_name.split('-')[0]
        
        # Create a subdirectory for the class if it doesn't exist
        class_dir = os.path.join(OUTPUT_DIR_ZCR, class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        
        # Extract the file name without extension
        filename = os.path.splitext(file_name)[0]
        
        # Check if the ZCR file already exists
        zcr_file_path = os.path.join(class_dir, f'{filename}_zcr.txt')
        if not os.path.exists(zcr_file_path):
            # Save the ZCR value to a text file in the class subdirectory
            with open(zcr_file_path, 'w') as f:
                f.write(f'ZCR: {zcr}')
    except Exception as e:
        print(f"An error occurred for {audio_file}: {e}")

def calculate_mfcc_and_save(audio_file, output_dir):
    try:
        # Extract the file name without extension
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Determine the class name from the audio file name
        class_name = filename.split('-')[0]

        # Check if the MFCC file already exists
        mfcc_file_path = os.path.join(OUTPUT_DIR_MFCC, class_name, f'{filename}_mfcc.npy')
        if not os.path.exists(mfcc_file_path):
            audio, sample_rate = librosa.load(audio_file, sr=None)  # Load the audio file
           
            # Calculate MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)  # You can adjust the number of MFCC coefficients

            # Create a subdirectory for the class if it doesn't exist
            class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            # Save the MFCCs to a binary file in the class subdirectory
            np.save(os.path.join(class_dir, f'{filename}_mfcc.npy'), mfccs)
    except Exception as e:
        print(f"An error occurred for {audio_file}: {e}")

def calculate_spectral_centroid_and_save(audio_file, output_dir):
    try:
        # Extract the file name without extension
        filename = os.path.splitext(os.path.basename(audio_file))[0]

        # Determine the class name from the audio file name
        class_name = filename.split('-')[0]

        # Check if the spectral centroid file already exists
        spec_centroid_file_path = os.path.join(output_dir, class_name, f'{filename}_spectral_centroid.npy')
        if not os.path.exists(spec_centroid_file_path):
            audio, sample_rate = librosa.load(audio_file, sr=None)  # Load the audio file

            # Calculate spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)

            # Create a subdirectory for the class if it doesn't exist
            class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            # Save the spectral centroid to a binary file in the class subdirectory
            np.save(os.path.join(class_dir, f'{filename}_spectral_centroid.npy'), spectral_centroid)
    except Exception as e:
        print(f"An error occurred for {audio_file}: {e}")

def calculate_spectral_bandwidth_and_save(audio_file, output_dir):
    try:
        # Extract the file name without extension
        filename = os.path.splitext(os.path.basename(audio_file))[0]

        # Determine the class name from the audio file name
        class_name = filename.split('-')[0]

        # Check if the spectral bandwidth file already exists
        spec_bandwidth_file_path = os.path.join(output_dir, class_name, f'{filename}_spectral_bandwidth.npy')
        if not os.path.exists(spec_bandwidth_file_path):
            audio, sample_rate = librosa.load(audio_file, sr=None)  # Load the audio file

            # Calculate spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)

            # Create a subdirectory for the class if it doesn't exist
            class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            # Save the spectral bandwidth to a binary file in the class subdirectory
            np.save(os.path.join(class_dir, f'{filename}_spectral_bandwidth.npy'), spectral_bandwidth)
    except Exception as e:
        print(f"An error occurred for {audio_file}: {e}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_ZCR, exist_ok=True)
os.makedirs(OUTPUT_DIR_MFCC, exist_ok=True)
os.makedirs(OUTPUT_DIR_SPECTRAL_COEF, exist_ok=True)
os.makedirs(OUTPUT_DIR_SPECTRAL_BW, exist_ok=True)
if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-logMel_spec')):
    os.mkdir(os.path.join(OUTPUT_DIR, 'audio-logMel_spec'))

for filename in os.listdir(INPUT_DIR):
    if "wav" in filename:
        file_path = os.path.join(INPUT_DIR, filename)
        calculate_zcr_and_save(file_path, OUTPUT_DIR_ZCR)
        calculate_mfcc_and_save(file_path, OUTPUT_DIR_MFCC)
        calculate_spectral_centroid_and_save(file_path, OUTPUT_DIR_SPECTRAL_COEF)
        calculate_spectral_bandwidth_and_save(file_path, OUTPUT_DIR_SPECTRAL_BW)
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