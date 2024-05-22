# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model('model_text_mfcc.h5')  # Provide the correct path to your model file

# # Load and preprocess the spectrogram image
# # Replace this with the actual path to your spectrogram PNG file
# spectrogram_image_path = r"D:\op\audio-images\airport-barcelona-0-0-a.png"

# # Load the spectrogram image and resize it to (64, 64)
# spectrogram_image = load_img(spectrogram_image_path, target_size=(64, 64))
# spectrogram_data = img_to_array(spectrogram_image)
# spectrogram_data = np.expand_dims(spectrogram_data, axis=0)  # Add batch dimension
# spectrogram_data = spectrogram_data / 255.0  # Normalize pixel values

# # Load and preprocess the MFCC data
# # Replace this with the actual path to your MFCC data
# mfcc_data = np.load(r"D:\op\mfcc_values\airport-barcelona-0-0-a_mfcc.npy")

# # Ensure that both data arrays have the same number of samples
# num_frames = 13
# num_mfcc_coefficients = 862
# mfcc_data = mfcc_data[:num_frames, :num_mfcc_coefficients]
# mfcc_data = mfcc_data.T
# mfcc_data=np.expand_dims(mfcc_data,axis=0)

# # Normalize the MFCC data (you may need to adjust the normalization based on your training data)
# mfcc_data = (mfcc_data - mfcc_data.mean()) / mfcc_data.std()

# # Load the ZCR value
# zcr_file_path = r"D:\op\zcr_values\airport-barcelona-0-0-a_zcr.txt"
# zcr_value = 0.0
# with open(zcr_file_path, 'r') as file:
#     # Read the first line and extract the numeric value
#     line = file.readline()
#     zcr_value = float(line.split(':')[1].strip())
# zcr_value=np.array([zcr_value])
# zcr_value=np.expand_dims(zcr_value,axis=0)
# # Predict the class using the loaded model
# class_probabilities = model.predict([spectrogram_data, mfcc_data,zcr_value])
# # Get the predicted class (index with the highest probability)
# predicted_class_index = np.argmax(class_probabilities)

# # Define your class names here (replace with your actual class names)
# class_names = ['class_airport', 'class_bus', 'class_metro', 'class_metro_station', 'class_park',
#                'class_public_square', 'class_shopping_mall', 'class_street_pedestrian', 'class_street_traffic', 'class_tram']

# # Get the predicted class name
# predicted_class_name = class_names[predicted_class_index]

# # Print the predicted class name or perform any further actions
# print(f'Predicted class: {predicted_class_name}')

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model_asc.hdf5')  # Provide the correct path to your model file

# Load the label encoder classes
label_encoder_classes = np.load('label_encoder_classes_asc.npy')

# Load and preprocess the spectrogram image
# Replace this with the actual path to your spectrogram PNG file
spectrogram_image_path = r"D:\output\audio-images\tram-vienna-285-8639-a.png"

# Load the spectrogram image and resize it to (64, 64)
spectrogram_image = load_img(spectrogram_image_path, target_size=(64, 64))
spectrogram_data = img_to_array(spectrogram_image)
spectrogram_data = np.expand_dims(spectrogram_data, axis=0)  # Add batch dimension
spectrogram_data = spectrogram_data / 255.0  # Normalize pixel values

# Load and preprocess the MFCC data
# Replace this with the actual path to your MFCC data
mfcc_data = np.load(r"D:\output\mfcc_values\tram-vienna-285-8639-a_mfcc.npy")

# Ensure that both data arrays have the same number of samples
num_frames = 13
num_mfcc_coefficients = 862
mfcc_data = mfcc_data[:num_frames, :num_mfcc_coefficients]
mfcc_data = mfcc_data.T
mfcc_data=np.expand_dims(mfcc_data,axis=0)

# Normalize the MFCC data (you may need to adjust the normalization based on your training data)
mfcc_data = (mfcc_data - mfcc_data.mean()) / mfcc_data.std()

# Load the ZCR value
zcr_file_path = r"D:\output\zcr_values\tram-vienna-285-8639-a_zcr.txt"
zcr_value = 0.0
with open(zcr_file_path, 'r') as file:
    # Read the first line and extract the numeric value
    line = file.readline()
    zcr_value = float(line.split(':')[1].strip())
zcr_value=np.array([zcr_value])
zcr_value=np.expand_dims(zcr_value,axis=0)
# Predict the class using the loaded model
predictions = model.predict([spectrogram_data, mfcc_data,zcr_value])
predicted_label = np.argmax(predictions, axis=1)
print(predicted_label)
predicted_class_label = label_encoder_classes[predicted_label[0]]
class_names = ['class_airport', 'class_bus', 'class_metro', 'class_metro_station', 'class_park',
               'class_public_square', 'class_shopping_mall', 'class_street_pedestrian', 'class_street_traffic', 'class_tram']
print(f"Predicted Class Label: {class_names[predicted_class_label]}")