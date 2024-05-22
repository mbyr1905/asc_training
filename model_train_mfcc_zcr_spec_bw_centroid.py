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
image_dir = "D:\\op_resampled\\audio-logMel_spec"
mfcc_dir = "D:\\op_resampled\\mfcc_values"
zcr_dir = "D:\\op_resampled\\zcr_values"
spec_bw_dir= "D:\\op_resampled\\spec_bw"
spec_centroid_dir = "D:\\op_resampled\\spec_centroid"

# Load image file names and corresponding class labels
image_files = os.listdir(image_dir)
#mfcc loading
mfcc_files = os.listdir(mfcc_dir)
#zcr loading 
zcr_files = os.listdir(zcr_dir)

spec_bw_files = os.listdir(spec_bw_dir)

spec_centroid_files = os.listdir(spec_centroid_dir)

# Initialize lists to store data
data = []
labels = []
mfcc_data = []
zcr_data=[]
spec_bw_data = []
spec_centroid_data = []

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
    with open(os.path.join(zcr_dir, image_file.replace(".png", "_zcr.txt"))) as zcr_file:
        zcr_string = zcr_file.read()
        zcr_value = float(zcr_string.split(": ")[1])
    zcr_data.append(zcr_value)
    sbw_values = np.load(os.path.join(spec_bw_dir, image_file.replace(".png", "_spectral_bandwidth.npy")))
    spec_bw_data.append(sbw_values)
    scc_values = np.load(os.path.join(spec_centroid_dir, image_file.replace(".png", "_spectral_centroid.npy")))
    spec_centroid_data.append(scc_values)
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
X_spec_bw = np.array(spec_bw_data)
X_spec_coeff = np.array(spec_centroid_data)

# Normalize the spectral bandwidth and coefficient data
X_spec_bw_mean = X_spec_bw.mean(axis=(0, 1))  # Calculate mean along the specified axes
X_spec_bw_std = X_spec_bw.std(axis=(0, 1))  # Calculate standard deviation along the specified axes

# Perform mean normalization on the spectral bandwidth data
X_spec_bw = (X_spec_bw - X_spec_bw_mean) / X_spec_bw_std

# Normalize the spectral coefficient data to a similar range
# Reshape the data if needed, assuming a range from 0 to 1
X_spec_coeff_min = X_spec_coeff.min()
X_spec_coeff_max = X_spec_coeff.max()

X_spec_coeff = (X_spec_coeff - X_spec_coeff_min) / (X_spec_coeff_max - X_spec_coeff_min)
# Normalize image data
X_img = X_img / 255.0
# Normalize the MFCC data (adjust the normalization as needed)
X_mfcc = (X_mfcc - X_mfcc.mean(axis=(0,1))) / X_mfcc.std(axis=(0,1))
X_zcr=np.array(zcr_data)

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 10

# Split the data into train and test sets
X_img_train, X_img_test, X_mfcc_train, X_mfcc_test, X_zcr_train, X_zcr_test, X_spec_bw_train, X_spec_bw_test, X_spec_coeff_train, X_spec_coeff_test, y_train, y_test = train_test_split(X_img, X_mfcc, X_zcr, X_spec_bw, X_spec_coeff, labels, test_size=0.2, random_state=42)

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

# Define the ZCR input
zcr_input = Input(shape=(1,), name="zcr_input")
z = Dense(64, activation='relu')(zcr_input)

# Define the input shape for Spectral Coefficient data
num_frames_spec_coeff =  1
num_spec_coefficients =  862
spec_coeff_input = Input(shape=(num_frames_spec_coeff, num_spec_coefficients, 1))

# Processing layers for spectral coefficients
c = Conv2D(32, (3, 3), padding='same', activation='relu')(spec_coeff_input)
c = BatchNormalization()(c)
c = MaxPooling2D(pool_size=(2, 2), padding='same')(c)
c = BatchNormalization()(c)
c = Conv2D(64, (3, 3), padding='same', activation='relu')(c)
c = BatchNormalization()(c)
c = MaxPooling2D(pool_size=(2, 2), padding='same')(c)
c = BatchNormalization()(c)
c = Conv2D(128, (3, 3), padding='same', activation='relu')(c)
c = BatchNormalization()(c)
c = MaxPooling2D(pool_size=(2, 2), padding='same')(c)
c = BatchNormalization()(c)
c = Flatten()(c)

# Define the input shape for Spectral Bandwidth data
num_frames_spec_bw =  1
num_spec_bw_values =  862
spec_bw_input = Input(shape=(num_frames_spec_bw, num_spec_bw_values, 1))

# Processing layers for spectral bandwidth
b = Conv2D(32, (3, 3), padding='same', activation='relu')(spec_bw_input)
b = BatchNormalization()(b)
b = MaxPooling2D(pool_size=(2, 2), padding='same')(b)
b = BatchNormalization()(b)
b = Conv2D(64, (3, 3), padding='same', activation='relu')(b)
b = BatchNormalization()(b)
b = MaxPooling2D(pool_size=(2, 2), padding='same')(b)
b = BatchNormalization()(b)
b = Conv2D(128, (3, 3), padding='same', activation='relu')(b)
b = BatchNormalization()(b)
b = MaxPooling2D(pool_size=(2, 2), padding='same')(b)
b = BatchNormalization()(b)
b = Flatten()(b)
# Concatenate all inputs
concatenated = Concatenate()([x, y, z, c, b])
m = Dense(128, activation="relu")(concatenated)
m = BatchNormalization()(m)
m = Dropout(0.5)(m)
output = Dense(num_classes, activation="softmax")(m)

model = Model(inputs=[image_input, mfcc_input, zcr_input, spec_coeff_input, spec_bw_input], outputs=output)

model.compile(
    loss='categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' if your labels are not one-hot encoded
    optimizer='adam',  # You can choose other optimizers like RMSprop or SGD
    metrics=['accuracy']  # Add any other metrics you want to track during training
)

# Training the model with all inputs (image, MFCC, ZCR, spectral bandwidth, spectral coefficient)
history = model.fit(
    [X_img_train, X_mfcc_train, X_zcr_train, X_spec_bw_train, X_spec_coeff_train],  
    to_categorical(y_train, num_classes=num_classes),  # Convert labels to categorical if needed
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=([X_img_test, X_mfcc_test, X_zcr_test, X_spec_bw_test, X_spec_coeff_test], to_categorical(y_test, num_classes=num_classes)),
    verbose=1  # You can adjust the verbosity level
)

model.save("model_text_mfcc_zcr_logmelspec_spec_bw_centroid_resampled.h5")

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