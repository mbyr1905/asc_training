from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model_text_mfcc.h5')  # Provide the correct path to your model file

# Load and preprocess the spectrogram image
# Replace this with the actual path to your spectrogram PNG file
spectrogram_image_path = r"D:\test_data_op\audio-images\class_metro\metro-vienna-228-6889-s6.png"

# Load the spectrogram image and resize it to (64, 64)
spectrogram_image = load_img(spectrogram_image_path, target_size=(64, 64))
spectrogram_data = img_to_array(spectrogram_image)
spectrogram_data = np.expand_dims(spectrogram_data, axis=0)  # Add batch dimension
spectrogram_data = spectrogram_data / 255.0  # Normalize pixel values

# Load and preprocess the MFCC data
# Replace this with the actual path to your MFCC data
mfcc_data = np.load(r"D:\test_data_op\mfcc_values\metro\metro-vienna-228-6889-s6_mfcc.npy")

# Ensure that both data arrays have the same number of samples
num_frames = 13
num_mfcc_coefficients = 862
mfcc_data = mfcc_data[:num_frames, :num_mfcc_coefficients]
mfcc_data = mfcc_data.T
mfcc_data=np.expand_dims(mfcc_data,axis=0)

# Normalize the MFCC data (you may need to adjust the normalization based on your training data)
mfcc_data = (mfcc_data - mfcc_data.mean()) / mfcc_data.std()

# Predict the class using the loaded model
class_probabilities = model.predict([spectrogram_data, mfcc_data])
# Get the predicted class (index with the highest probability)
predicted_class_index = np.argmax(class_probabilities)

# Define your class names here (replace with your actual class names)
class_names = ['class_airport', 'class_bus', 'class_metro', 'class_metro_station', 'class_park',
               'class_public_square', 'class_shopping_mall', 'class_street_pedestrian', 'class_street_traffic', 'class_tram']

# Get the predicted class name
predicted_class_name = class_names[predicted_class_index]

# Print the predicted class name or perform any further actions
print(f'Predicted class: {predicted_class_name}')
