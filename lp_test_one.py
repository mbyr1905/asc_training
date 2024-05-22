from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Path to the directory containing images of individuals
dataset_path = r"C:\Users\Bhanu2003\Downloads\dataset\dataset"
num_epochs = 10  # Number of training epochs
batch_size = 32  # Batch size for training
validation_split = 0.2  # Fraction of data to use for validation

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom output layer for classification
folders = sorted(os.listdir(dataset_path))
num_classes = len(folders)
global_average_layer = GlobalAveragePooling2D()(base_model.output)
output = Dense(num_classes, activation='softmax')(global_average_layer)

# Create a new model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initialize empty lists to store image paths and labels
image_paths = []
labels = []

# Iterate over each label and corresponding folder
for label, folder in enumerate(folders):
    # Get list of images in the current folder
    images = os.listdir(os.path.join(dataset_path, folder))
    # Append full paths of images to image_paths list
    image_paths.extend([os.path.join(dataset_path, folder, img) for img in images])
    # Append label to labels list for each image in the folder
    labels.extend([label] * len(images))


# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, labels, test_size=validation_split, random_state=42)


# Create data generators for training and validation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)



valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



train_generator = train_datagen.flow_from_directory(
    directory=dataset_path,
    classes=folders,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    seed=42
)



valid_generator = valid_datagen.flow_from_directory(
    directory=dataset_path,
    classes=folders,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=False,
    seed=42
)


# Train the model
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=valid_generator
)

# Save the model
model.save('lp_face_recognition_model_vggface.h5')

# Save the class labels
np.save('lp_class_labels.npy', train_generator.class_indices)