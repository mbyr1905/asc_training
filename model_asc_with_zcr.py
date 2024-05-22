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

# Define the prepare function
def prepare(ds, augment=False):
    # Define our transformations
    rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])
    flip_and_rotate = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    
    # Apply rescale to both datasets and augmentation only to training
    ds = ds.map(lambda x, y: (rescale(x, training=True), y))
    if augment:
        ds = ds.map(lambda x, y: (flip_and_rotate(x, training=True), y))
    return ds


# Set paths to input and output data
INPUT_DIR = 'D:\input'
OUTPUT_DIR = 'D:\output'
OUTPUT_DIR_ZCR = 'D:\output\zcr_values'

# ... (rest of your code)
parent_list = os.listdir(INPUT_DIR)
# Print the ten classes in our dataset
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

# For every recording, make a spectogram and save it as label_speaker_no.png
if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-images')):
    os.mkdir(os.path.join(OUTPUT_DIR, 'audio-images'))
if not os.path.exists(os.path.join(OUTPUT_DIR, 'zcr_values')):
    os.mkdir(os.path.join(OUTPUT_DIR, 'zcr_values'))

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

for filename in os.listdir(INPUT_DIR):
    if "wav" in filename:
        file_path = os.path.join(INPUT_DIR, filename)
        calculate_zcr_and_save(file_path, OUTPUT_DIR_ZCR)
        file_stem = Path(file_path).stem
        target_dir = f'class_{file_stem.split("-",1)[0]}'
        dist_dir = os.path.join(os.path.join(OUTPUT_DIR, 'audio-images'), target_dir)
        file_dist_path = os.path.join(dist_dir, file_stem)
        if not os.path.exists(file_dist_path + '.png'):
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)
            file_stem = Path(file_path).stem
            sound_info, frame_rate = get_wav_info(file_path)
            pylab.specgram(sound_info, Fs=frame_rate)
            pylab.savefig(f'{file_dist_path}.png')
            pylab.close()

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 64
N_CHANNELS = 3
N_CLASSES = 10

# Make a dataset containing the training spectrograms
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=os.path.join(OUTPUT_DIR, 'audio-images'),
                                             shuffle=True,
                                             color_mode='rgb',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="training",
                                             seed=0)

# Make a dataset containing the validation spectrograms
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=os.path.join(OUTPUT_DIR, 'audio-images'),
                                             shuffle=True,
                                             color_mode='rgb',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="validation",
                                             seed=0)
plt.figure(figsize=(12, 12))
for images, labels in train_dataset.take(1):
    for i in range(min(9,len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

train_dataset = prepare(train_dataset, augment=False)
valid_dataset = prepare(valid_dataset, augment=False)

# Create CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)))
model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))

# Create F1-score metric
f1_score_metric = tf.keras.metrics.F1Score(average='weighted')

# Compile model with F1-score metric
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],  # Use accuracy as the metric for training
)

# Train model for 10 epochs, capture the history
history = model.fit(train_dataset, epochs=20, validation_data=valid_dataset)
model.save('trained_audio_classification_model_one64.h5')

# Plot the F1-score curves for training and validation.
history_dict = history.history
accuracy_values = history_dict['accuracy']
val_accuracy_values = history_dict['val_accuracy']
epochs = range(1, len(accuracy_values) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy_values, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy_values, 'b', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Compute the final accuracy
final_accuracy = accuracy_values[-1]
print("Final Accuracy: {0:.6f}".format(final_accuracy))