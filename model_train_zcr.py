import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import F1Score
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import wave
import pylab
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools
import shutil
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

# Define paths to your image and ZCR directories
image_dir = "D:\\op\\audio-images"
zcr_dir = "D:\\op\\zcr_values"
mfcc_dir = "D:\\op\\mfcc_values"

# Load image file names and corresponding class labels
image_files = os.listdir(image_dir)
# zcr loading
zcr_files = os.listdir(zcr_dir)
#mfcc loading
mfcc_files = os.listdir(mfcc_dir)

# Initialize lists to store data
data = []
labels = []

# Loop through the files to combine data
for image_file in image_files:
    # Extract the class label from the image file name
    class_label = image_file.split("-")[0]
    
    # Load the spectrogram image and ZCR value
    image = load_img(os.path.join(image_dir, image_file), target_size=(64, 64))  # Adjust target_size as needed
    image = img_to_array(image)
    with open(os.path.join(zcr_dir, image_file.replace(".png", "_zcr.txt"))) as zcr_file:
        zcr_string = zcr_file.read()
        zcr_value = float(zcr_string.split(": ")[1])
    # Append data and labels
    data.append((image, zcr_value))
    labels.append(class_label)

# Encode class labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split data into training and validation sets
X_img = np.array([item[0] for item in data])  # Extract image data
X_zcr = np.array([item[1] for item in data])  # Extract ZCR values

# Normalize image data
X_img = X_img / 255.0

# Concatenate the image data and ZCR values
#X_combined = np.concatenate((X_img, X_zcr.reshape(-1, 1)), axis=1)
X_train_img, X_val_img, X_train_zcr, X_val_zcr, Y_train, Y_val = train_test_split(X_img, X_zcr, labels, test_size=0.2, random_state=42)


# One-hot encode the labels
Y_train = to_categorical(Y_train, num_classes)
Y_val = to_categorical(Y_val, num_classes)

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 10

# Define the input layers for image and ZCR data
input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS))
input_zcr = Input(shape=(1,))

# Model for image data
x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(128, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Flatten()(x)

# Model for ZCR data
y = Dense(64, activation='relu')(input_zcr)

# Combine the two models
combined = tf.keras.layers.concatenate([x, y])
z = Dense(256, activation='relu')(combined)
z = BatchNormalization()(z)
z = Dropout(0.5)(z)
output = Dense(N_CLASSES, activation='softmax')(z)

# Create the model
model = Model(inputs=[input_img, input_zcr], outputs=output)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy']
)

history = model.fit(
    [X_train_img, X_train_zcr], Y_train,
    validation_data=([X_val_img, X_val_zcr], Y_val),
    epochs=10,
    batch_size=32  # You can adjust the batch size as needed
)

model.save("model_text_zcr_one.h5")

# Plot the accuracy and validation accuracy
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
final_val_accuracy = val_accuracy_values[-1]
print("Final Training Accuracy: {0:.6f}".format(final_accuracy))
print("Final Validation Accuracy: {0:.6f}".format(final_val_accuracy))