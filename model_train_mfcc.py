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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Reshape  # Add this import
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf

def f1_metric(y_true, y_pred):
    y_pred = np.round(y_pred)
    return tf.py_function(lambda: f1_score(y_true, y_pred, average='micro'), [], tf.float32)

# Define paths to your image and ZCR directories
image_dir = "D:\\op\\audio-images"
mfcc_dir = "D:\\op\\mfcc_values"

# Load image file names and corresponding class labels
image_files = os.listdir(image_dir)
#mfcc loading
mfcc_files = os.listdir(mfcc_dir)

# Initialize lists to store data
data = []
labels = []
mfcc_data = []

# Loop through the files to combine data
for image_file in image_files:
    # Extract the class label from the image file name
    class_label = image_file.split("-")[0]
    
    # Load the spectrogram image and ZCR value
    image = load_img(os.path.join(image_dir, image_file), target_size=(64, 64))  # Adjust target_size as needed
    image = img_to_array(image)
    mfcc_values = np.load(os.path.join(mfcc_dir, image_file.replace(".png", "_mfcc.npy")))
    mfcc_data.append(mfcc_values)
    # Append data and labels
    data.append(image)
    labels.append(class_label)

# Encode class labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split data into training and validation sets
X_img = np.array(data)
X_mfcc=np.array(mfcc_data)
X_mfcc = np.transpose(X_mfcc, (0, 2, 1))  # Transpose the dimensions
X_mfcc = X_mfcc[..., np.newaxis]  # Add a new dimension for channels (1)

# print(X_mfcc.shape)
# Normalize image data
X_img = X_img / 255.0
# Normalize the MFCC data (adjust the normalization as needed)
X_mfcc = (X_mfcc - X_mfcc.mean(axis=(0,1))) / X_mfcc.std(axis=(0,1))


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 10

image_input = Input(shape=(64, 64, 3))  # Adjust the input shape as needed

x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(image_input)
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

num_frames = 862
num_mfcc_coefficients = 13
# Define the input shape for MFCC data
mfcc_input = Input(shape=(num_frames, num_mfcc_coefficients, 1))

y = Conv2D(32, (3, 3), padding='same', activation='relu')(mfcc_input)
y = BatchNormalization()(y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = BatchNormalization()(y)
y = Conv2D(64, (3, 3), padding='same', activation='relu')(y)
y = BatchNormalization()(y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = BatchNormalization()(y)
y = Conv2D(128, (3, 3), padding='same', activation='relu')(y)
y = BatchNormalization()(y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = BatchNormalization()(y)
y = Flatten()(y)

merged = Concatenate()([x, y])

# Add fully connected layers and output layer
dense1 = Dense(128, activation='relu')(merged)
output = Dense(num_classes, activation='softmax')(dense1)

# Create the model
model = Model(inputs=[image_input, mfcc_input], outputs=output)

model.compile(
    loss='categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' if your labels are not one-hot encoded
    optimizer='adam',  # You can choose other optimizers like RMSprop or SGD
    metrics=['accuracy']  # Add any other metrics you want to track during training
)

# Split your data into training and validation sets
X_train_img, X_val_img, X_train_mfcc, X_val_mfcc, y_train, y_val = train_test_split(X_img, X_mfcc, labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

# Train the model
history = model.fit(
    [X_train_img, X_train_mfcc],  # Provide both image and MFCC data
    y_train,  # Labels
    batch_size=BATCH_SIZE,
    epochs=10,
    validation_data=([X_val_img, X_val_mfcc], y_val),  # Validation data
    verbose=1  # You can adjust the verbosity level
)

model.save("model_test_mfcc_one.h5")

# Print the training and validation accuracy
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")