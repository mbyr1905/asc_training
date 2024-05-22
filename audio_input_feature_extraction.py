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
OUTPUT_DIR = 'D:\\output'
OUTPUT_DIR_ZCR = 'D:\\output\\zcr_values'
OUTPUT_DIR_MFCC='D:\\output\\mfcc_values'
# OUTPUT_DIR_CQT = 'D:\\test_data_op\\cqt_values'

def get_mel_spectrogram(wav_file):
    try:
        y, sr = librosa.load(wav_file, sr=None) 
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr) 
        return mel_spectrogram
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_wav_info(wav_file):
    try:
        wav = wave.open(wav_file, 'r')
        frames = wav.readframes(-1)
        sound_info = pylab.frombuffer(frames, 'int16')
        frame_rate = wav.getframerate()
        wav.close()
        return sound_info, frame_rate
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_ZCR, exist_ok=True)
os.makedirs(OUTPUT_DIR_MFCC, exist_ok=True)

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

# Function to calculate MFCCs and save results
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


for filename in os.listdir(INPUT_DIR):
    if "wav" in filename:
        file_path = os.path.join(INPUT_DIR, filename)
        calculate_zcr_and_save(file_path, OUTPUT_DIR_ZCR)
        calculate_mfcc_and_save(file_path, OUTPUT_DIR_MFCC)
        file_stem = Path(file_path).stem
        target_dir = f'class_{file_stem.split("-",1)[0]}'
        dist_dir = os.path.join(os.path.join(OUTPUT_DIR, 'audio-images'), target_dir)
        file_dist_path = os.path.join(dist_dir, file_stem)
        if not os.path.exists(file_dist_path + '.png'):
            os.makedirs(dist_dir, exist_ok=True)
            file_stem = Path(file_path).stem
             # Load the audio file
            y, sr = librosa.load(file_path, sr=None)
            
            # Calculate the mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

            # Convert to decibels
            log_mel_spec = librosa.power_to_db(mel_spectrogram, ref=np.max)
            plt.figure(figsize=(8, 6))
            librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Spectrogram for {os.path.basename(file_path)}")
            plt.savefig(f'{file_dist_path}.png')
            plt.close()
            # sound_info, frame_rate = get_mel_spectrogram(file_path)
            # pylab.specgram(sound_info, Fs=frame_rate)
            # pylab.savefig(f'{file_dist_path}.png')
            # pylab.close()
# for filename in os.listdir(INPUT_DIR):
#     if "wav" in filename:
#         file_path = os.path.join(INPUT_DIR, filename)
#         file_stem = Path(file_path).stem
#         target_dir = f'class_{file_stem.split("-",1)[0]}'
#         dist_dir = os.path.join(os.path.join(OUTPUT_DIR, 'audio-images_1'), target_dir)
#         file_dist_path = os.path.join(dist_dir, file_stem)
#         if not os.path.exists(file_dist_path + '.png'):
#             if not os.path.exists(dist_dir):
#                 os.mkdir(dist_dir)
#             file_stem = Path(file_path).stem
#             mel_spectrogram = get_mel_spectrogram(file_path)  # Get the mel spectrogram
#             plt.figure(figsize=(10, 4))
#             librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
#             plt.colorbar(format='%+2.0f dB')
#             plt.title('Mel Spectrogram')
#             plt.tight_layout()
#             plt.savefig(f'{file_dist_path}.png')
#             plt.close()

print("done extracting")