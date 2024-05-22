import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder

# Load the saved model and label encodings
model = load_model(r"C:\Users\Bhanu2003\OneDrive\Desktop\final_year_project\project_1\lp_face_recognition_model_vggface.h5")
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(r"C:\Users\Bhanu2003\OneDrive\Desktop\final_year_project\project_1\lp_class_labels.npy", allow_pickle=True)

# Print the contents of the loaded array for debugging
# print("Label encodings:", label_encoder.classes_)

# Function to detect faces using MTCNN and resize them
def detect_faces_and_resize(image, target_size=(224, 224)):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    if faces is None:
        return []  # Return an empty list if no faces are detected
    resized_faces = []
    for face in faces:
        x, y, w, h = face['box']
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, target_size)
        resized_faces.append(face_img)
    return resized_faces

# Function to make predictions for each detected face
def predict_faces(faces):
    predictions = []
    for face in faces:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        prediction = model.predict(face)
        max_prob = np.max(prediction)
        if max_prob >= 0.2:
            class_idx = np.argmax(prediction)
            try:
                s = str(temp[class_idx-1])
                predictions.append(s)
            except IndexError:
                print("IndexError occurred. class_idx:", class_idx, "label_encoder.classes_:", label_encoder.classes_)
    return predictions


# Function to recognize faces in a group photo
def recognize_faces_in_photo(photo_path):
    image = cv2.imread(photo_path)
    faces = detect_faces_and_resize(image)
    recognized_faces = predict_faces(faces)
    return recognized_faces
temp = ['Aravind', 'Bhanu Yaswanth', 'Bhargav', 'Brahmesh', 'Chakradhar', 'Chandra Shekhar', 'DeviSri', 'Dhanush Eti', 'Dheeraj B', 'Gautham Datta', 'Gautham Sai', 'Gayatri', 'Harinath', 'Ishratalikhan', 'Jaivanth', 'Kasi Viswanath', 'Lekesh', 'Lokeshwar', 'Mahir', 'Manoj Kumar', 'Mohit varma', 'Nida', 'P Harshit', 'Phanindra', 'Pranav', 'Rajmouli', 'Rishendra', 'Rohan', 'Rohit J', 'Roshan', 'Sidhu', 'Snehith', 'Sruthilaya', 'Tarun', 'Teja', 'Umesh', 'nitin', 'p. varun kumar']

photo_path = 'group_image_complete.jpg'
recognized_faces = recognize_faces_in_photo(photo_path)
print('Recognized faces:', recognized_faces)
