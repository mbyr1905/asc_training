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
from tensorflow.keras.utils import to_categorical

# Path to the directory containing images of individuals
dataset_path = r"D:\train"
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
output = Dense(num_classes, activation='softmax')(base_model.output)

# Create a new model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

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

# Convert lists to numpy arrays
X_train = np.array(image_paths)
y_train = np.array(labels)

# Convert labels to one-hot encoded format
y_train_onehot = to_categorical(y_train, num_classes=num_classes)

# Create DataFrame using the image paths and labels
train_df = pd.DataFrame({'image_path': X_train, 'label': y_train})

# Convert integer labels to strings
train_df['label'] = train_df['label'].astype(str)

# Splitting data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(train_df['image_path'], train_df['label'], test_size=validation_split, random_state=42)

# Create data generators for training and validation
train_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
).flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='label',  # Note: Change this to None
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='input',  # Change class_mode to 'input'
    dtype='str'
)

valid_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_dataframe(
    dataframe=pd.DataFrame({'image_path': X_valid, 'label': y_valid}),
    x_col='image_path',
    y_col='label',  # Note: Change this to None
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='input'  # Change class_mode to 'input'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_generator)
)

# Save the model
model.save('face_recognition_model_vggface.h5')

# Create and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(train_df['label'])

# Save the label encoder in npy format
np.save('label_encoder.npy', label_encoder.classes_)