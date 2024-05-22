# import numpy as np
# from tensorflow.keras.models import load_model
# import librosa
# import matplotlib.pyplot as plt
# import librosa.display
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load the trained model
# model = load_model("model_asc.hdf5")

# # Load the label encoder classes
# label_encoder_classes = np.load('label_encoder_classes_asc.npy')

# # Function to preprocess audio sample and extract features
# def preprocess_and_extract_features(audio_path):
#     try:
#         # Calculate ZCR
#         y, sr = librosa.load(audio_path, sr=None)
#         zcr = np.mean(np.abs(np.diff(np.sign(y))) / 2.0)
        
#         # Calculate MFCCs
#         audio, sample_rate = librosa.load(audio_path, sr=None)
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        
#         # Reshape MFCCs to match the input shape of the MFCC layer
#         mfccs = mfccs[..., np.newaxis]

#         # Calculate mel spectrogram
#         mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate)
#         log_mel_spec = librosa.power_to_db(mel_spectrogram, ref=np.max)

#         # Create an image array
#         plt.figure(figsize=(8, 6))
#         librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
#         plt.colorbar(format='%+2.0f dB')
#         plt.close()
        
#         image_array = img_to_array(plt.gcf().get_axes()[0].images[0])
#         image_array = image_array[np.newaxis, ...] / 255.0

#         return image_array, zcr, mfccs
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None, None, None

# # Replace this path with the path to your audio file
# audio_path = r"D:\input\audio\park-lisbon-1052-41218-a.wav"

# # Preprocess and extract features from the audio sample
# image_array, zcr_value, mfcc_values = preprocess_and_extract_features(audio_path)
# # Check if the features are valid
# if image_array is not None and mfcc_values is not None:
#     # Make predictions
#     predictions = model.predict([image_array, mfcc_values, np.array([zcr_value])])

#     # Get the predicted class label
#     predicted_class_index = np.argmax(predictions)
#     predicted_class_label = label_encoder_classes[predicted_class_index]

#     print(f"Predicted Class Label: {predicted_class_label}")
# else:
#     print("Feature extraction failed. Please check the audio file and the feature extraction function.")

import numpy as np
from tensorflow.keras.models import load_model
import librosa
from tensorflow.keras.preprocessing.image import img_to_array
import librosa.display
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('model_asc.hdf5')  # Provide the correct path to your model file

# Load the label encoder classes
label_encoder_classes = np.load('label_encoder_classes_asc.npy')

# Function to preprocess audio sample and extract features
def preprocess_and_extract_features(audio_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Calculate ZCR
        zcr = np.mean(np.abs(np.diff(np.sign(y))) / 2.0)

        # Calculate MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Calculate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        log_mel_spec = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Create an image array
        plt.figure(figsize=(8, 6))
        librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.close()

        spectrogram_image = img_to_array(plt.gcf().get_axes()[0].images[0])
        spectrogram_image = spectrogram_image[np.newaxis, ...] / 255.0

        # Ensure that MFCCs have the correct shape
        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 862 - mfccs.shape[1]))), mode='constant')

        return spectrogram_image, zcr, mfccs
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None
    
# Replace this path with the path to your audio file
audio_path = r"D:\input\audio\park-lisbon-1052-41218-a.wav"

# Preprocess and extract features from the audio sample
spectrogram_image, zcr_value, mfcc_values = preprocess_and_extract_features(audio_path)

# Check if the features are valid
if spectrogram_image is not None and mfcc_values is not None:
    # Make predictions
    predictions = model.predict([spectrogram_image, mfcc_values, np.array([zcr_value])])

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = label_encoder_classes[predicted_class_index]

    print(f"Predicted Class Label: {predicted_class_label}")
else:
    print("Feature extraction failed. Please check the audio file and the feature extraction function.")
