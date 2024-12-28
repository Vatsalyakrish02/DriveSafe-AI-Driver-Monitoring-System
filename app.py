from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer
import tensorflow_hub as hub  # Import TensorFlow Hub for custom KerasLayer
import time  # For cooldown mechanism

# Initialize the Flask app
app = Flask(__name__)

# Model path
MODEL_PATH = "C:/Users/vatsa/Downloads/Driver_drowsiness_detection/model/20241227-2054-full-image-set-mobilenetv2-Adam.h5"

# Load model with the custom KerasLayer
model = load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

# Labels for the model's output
labels = ['Closed', 'Open']

# Initialize pygame mixer for sound
mixer.init()
beep_sound = mixer.Sound("C:/Users/vatsa/Downloads/Driver_drowsiness_detection/alarm.wav")  # Replace with the path to your beep sound file

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cooldown variables for the beep sound
last_alert_time = 0
alert_cooldown = 2  # seconds

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (224, 224))  # Resize to match the input size of MobilenetV2
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to detect face and return the region of interest
def detect_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        return frame[y:y+h, x:x+w]  # Return the first detected face
    return None

# Video capture generator
def generate_frames():
    cap = cv2.VideoCapture(0)

    # Set frame size for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    global last_alert_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect ROI dynamically using face detection
        roi = detect_roi(frame)
        if roi is not None:
            # Preprocess the region of interest
            processed_roi = preprocess_image(roi)

            try:
                # Predict the state of the eyes
                prediction = model.predict(processed_roi)
                state = labels[np.argmax(prediction)]

                # Display the prediction on the frame
                cv2.putText(frame, f"State: {state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Trigger beep if eyes are closed and cooldown has passed
                if state == 'Closed' and (time.time() - last_alert_time) > alert_cooldown:
                    beep_sound.play()
                    last_alert_time = time.time()

            except Exception as e:
                cv2.putText(frame, "Prediction Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Prediction failed: {e}")

        else:
            # Indicate no face was detected
            cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
