
# Sign Language & ASL Recognition System

This project provides a real-time sign language and American Sign Language (ASL) recognition system using a combination of TensorFlow, PyTorch, and MediaPipe for hand landmark and holistic pose detection. It utilizes a custom LSTM model for sequence-based predictions and a hand landmark classifier for recognizing individual ASL signs.

## Features

- **TensorFlow Model**: Recognizes sequences of sign language gestures using a custom LSTM.
- **PyTorch Model**: Classifies individual hand landmarks into ASL alphabets and commands.
- **MediaPipe Integration**: Detects and processes face, pose, and hand landmarks.
- **Real-time Webcam Input**: Uses a webcam to capture gestures and make predictions.
- **Confidence Threshold**: Only accepts predictions above a certain confidence threshold.

## Requirements

- Python 3.x
- TensorFlow (2.x)
- PyTorch
- OpenCV
- MediaPipe
- NumPy

To install the required dependencies, run:

bash
pip install tensorflow torch opencv-python mediapipe numpy


## Setup Instructions
# Prepare the models:

--**Download** the TensorFlow model (action.h5) for gesture sequence prediction.
--**Download** the PyTorch model (asl_landmark_model.pth) for hand landmark classification.
--**Running the application:** After setting up the models and installing the dependencies, run the script:

```bash
python sign_language_recognition.py
```

# The system will use your webcam to detect gestures in real-time, predict sign language sequences, and identify individual ASL gestures using hand landmarks.

# Workflow
*Frame Capture:* Captures video frames from the webcam.
*Pose and Hand Landmark Detection:* Uses MediaPipe to detect body, face, and hand landmarks.
**Gesture Recognition:**
*TensorFlow Model:* Recognizes sequences of hand gestures (e.g., "hello", "thanks", "iloveyou").
*PyTorch Model:* Classifies hand landmarks into ASL alphabet letters and commands (e.g., 'A', 'B', 'C', 'space', 'del').
*Real-time Prediction:* Displays the predicted gesture or sign on the screen, along with confidence scores.
*FPS Display:* Shows the current frames per second (FPS) for performance monitoring.


Explanation of Key Components
1. Custom LSTM in TensorFlow
The TensorFlow model uses a custom LSTM layer to handle sequence-based predictions. The CustomLSTM class removes the unsupported time_major argument during model loading.

2. Hand Landmark Classifier in PyTorch
The PyTorch model classifies individual hand landmarks into 29 ASL symbols. The landmarks are processed by the model to provide predictions, with a threshold of 70% confidence to accept the classification.

3. MediaPipe for Hand and Pose Detection
MediaPipe is used to extract key points from the body, face, and hands, which are passed into the TensorFlow and PyTorch models for recognition.

4. Frame Processing & Prediction Display
The system processes each frame from the webcam, detects landmarks, and makes predictions for gesture sequences and hand landmarks. The predicted labels and confidence are displayed on the screen.

Example Outputs
TensorFlow Model Prediction:
Displayed as "Seq: [Gesture Name] ([Confidence])"
Example: "Seq: HELLO (0.95)"
PyTorch Model Prediction:
Displayed as "Hand: [Sign] ([Confidence])"
Example: "Hand: A (0.85)"
Key Settings
Confidence Threshold for PyTorch Model: The system uses a confidence threshold (CONFIDENCE_THRESHOLD = 0.7) for hand landmark classification.
Sequence Length for TensorFlow Model: The system analyzes 30 frames at a time for gesture sequence prediction.
Troubleshooting
If the webcam isn't opening, ensure that the webcam is properly connected and no other application is using it.
If predictions are not accurate, consider training the models with more data or fine-tuning the existing models.
Ensure that the model paths (action.h5 and asl_landmark_model.pth) are correctly specified.


### **Note**: some issues are still present, in the future i will retrain all the models under the same architecture, or maybe find an nlp solution, hope you enjoy it, there is another repo where i explain how i trained the dataset step by step. 


enjoy :)
