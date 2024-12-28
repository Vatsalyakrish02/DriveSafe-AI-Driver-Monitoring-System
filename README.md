# DriveSafe AI: Driver Monitoring System

A real-time drowsiness detection system utilizing deep learning and computer vision to enhance road safety. The system identifies whether a driver's eyes are open or closed and triggers an alert if drowsiness is detected.

## Features

Real-time monitoring using a webcam.
AI-driven eye state classification (Open/Closed) based on a fine-tuned MobileNetV2 model.
Audio alert system to warn drivers in case of detected drowsiness.
Flask web application for seamless integration and user interface.

## Tech Stack

* Python: Backend logic and integration.
* TensorFlow: Deep learning framework for model training and inference.
* Flask: Web framework for real-time video streaming.
* OpenCV: Image processing and video frame capture.
* Pygame: Audio playback for the alert system.
* TensorFlow Hub: Pre-trained model for transfer learning.

## Installation

**Prerequisites**

* Python 3.7 or above
* Virtual environment setup (optional but recommended)

Steps

1. Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/drowsiness-detection-system.git  
cd drowsiness-detection-system  

2. Install dependencies:

bash
Copy code
pip install -r requirements.txt  

3. Place your trained model in the /model directory (update the file path in app.py if needed).

4. Place the audio alert file (alarm.wav) in the /sounds directory.

## Usage

1. Run the Flask application:

bash
Copy code
python app.py  
Open a web browser and navigate to:

2. Open a web browser and navigate to:

Copy code
http://127.0.0.1:5000/  
Allow access to your webcam, and the application will start monitoring your eyes.

## Directory Structure

perl
Copy code
drowsiness-detection-system/  
│  
├── app.py                     # Flask application script  
├── PreparedData               # Directory containing images for training and testing

├── customtesting              # Directory containing images for testing

├── logs                       # Directory contains tensorboard log details
├── requirements.txt           # Required Python libraries

├── model/                     # Directory for the trained model  
├── alarm.wav                  # Alert for audio files  
├── templates/                 # HTML templates for Flask  
│   └── index.html             # Web interface  
├── drowsiness_detection.ipynb # Jupyter notebook file
├── image_labels.csv           # CSV file for image labels
├── License                    # License for the github repository  
└── README.md                  # Project documentation  

## Model Details

* Architecture: MobileNetV2 fine-tuned for binary classification (Open/Closed).
* Training Dataset: OpenNed Closed Eyes Dataset

## Future Improvements

* Enhancing model accuracy with diverse datasets.
* Adding multi-feature detection like yawning or head tilts.
* Optimizing for deployment on edge devices (e.g., Raspberry Pi).

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

