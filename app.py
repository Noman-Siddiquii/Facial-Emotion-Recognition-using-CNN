import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("cnn_model.h5")

# Emotion labels
emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocess face for prediction
def preprocess_face(face_img):
    resized = cv2.resize(face_img, (100, 100))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 100, 100, 3))
    return reshaped

# Predict emotion and return label + confidence
def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    preds = model.predict(processed)[0]
    emotion_idx = np.argmax(preds)
    emotion_label = emotion_labels[emotion_idx]
    confidence = preds[emotion_idx]
    return emotion_label, confidence

# Annotate frame with bounding boxes and emotion
def annotate_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(face)
        label = f"{emotion} ({confidence*100:.2f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# App title
st.title("Real-time Facial Emotion Recognition")

# Webcam toggle button with session state
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False

def toggle_webcam():
    st.session_state.run_webcam = not st.session_state.run_webcam

button_label = "Stop Webcam" if st.session_state.run_webcam else "Start Webcam"
st.button(button_label, on_click=toggle_webcam)

# Webcam frame display
stframe = st.empty()

if st.session_state.run_webcam:
    cap = cv2.VideoCapture(0)

    try:
        while st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from webcam.")
                break

            annotated_frame = annotate_frame(frame)
            stframe.image(annotated_frame, channels="RGB")
    finally:
        cap.release()
