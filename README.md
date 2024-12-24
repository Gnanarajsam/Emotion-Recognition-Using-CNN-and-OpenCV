
# Emotion Recognition Using CNN and OpenCV

This project leverages deep learning and computer vision to detect and classify human emotions from real-time video streams or images. By utilizing Convolutional Neural Networks (CNNs) for feature extraction and OpenCV for face detection, this system analyzes facial expressions and categorizes them into predefined emotional states.

## Features
• Real-Time Emotion Detection: Analyze emotions from live video streams.

• Robust Face Detection: Utilize Haar cascades for accurate face detection.

• Deep Learning Model: Employ CNN-based architectures for emotion classification.

• Dataset Integration: Prepares data for training and testing using well-known datasets.

• Pre-trained Models: Use pre-trained weights to improve accuracy and efficiency.

## Dataset
###  FER 2013
The Facial Expression Recognition (FER) 2013 dataset is used for training and evaluation. It is a publicly available dataset provided by Kaggle.

#### Dataset Features:

Size: 35,887 grayscale images, each of size 48x48 pixels.

Classes: Seven categories of emotions:

    Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

#### Structure:
Training set: ~80% of images.
Test set: ~20% of images.
Source:
FER 2013 on Kaggle

## Setup Instructions
### 1. Prerequisites
Ensure you have Python installed on your system (recommended: Python 3.8 or above).

### 3. Install Required Libraries
Install the dependencies using the following command:

    pip install -r requirements.txt

### 4. Model and Cascade Files
Download the following files and place them in the appropriate directories:

Haar Cascade File for Face Detection: haarcascade_frontalface_default.xml

Directory: haarcascade_files/

Pre-trained Model: _mini_XCEPTION.102-0.66.hdf5

Directory: models/
### 5. Dataset Preparation
Download the FER 2013 dataset from Kaggle and place it in the dataset/ folder. Use load_and_process.py to preprocess the dataset for training and testing.

## Usage
### 1. Run the Real-Time Emotion Recognition
Execute the following command to start the application:

    python real_time_video.py
### 2. File Structure
Copy code
emotion-recognition/

├── haarcascade_files/

    └── haarcascade_frontalface_default.xml


├── models/

    └── _mini_XCEPTION.102-0.66.hdf5


├── dataset/

    └── fer2013.csv


├── real_time_video.py

├── cnn.py

├── load_and_process.py

├── requirements.txt

## Important Libraries
### 1. Feature Extraction
OpenCV:
Used for image preprocessing, face detection, and capturing real-time video streams.
### Installation:
    pip install keras, numpy, pandas, opencv-python, imutils, scikit-learn, tensorflow, h5py

### Key libraries in the project:
• TensorFlow/Keras: For building and training the CNN model.

• OpenCV: For real-time face detection and preprocessing.

• NumPy: For numerical computations.

• h5py: To handle .hdf5 files for saving/loading trained models.

## Key Concepts
### 1. Feature Extraction
Facial features are extracted using Haar Cascade Classifiers provided by OpenCV.

### 2. Convolutional Neural Networks (CNN)
The CNN model analyzes facial expressions and predicts emotions using a pre-trained _mini_XCEPTION architecture.

### 3. Dataset Handling
The dataset is preprocessed into train-test splits, resized, and normalized for efficient model training and evaluation.

## Acknowledgments
• OpenCV for face detection.

• Keras and TensorFlow for building and deploying deep learning models.

• The FER 2013 dataset for training and evaluation.

• The pre-trained _mini_XCEPTION model for emotion classification.








