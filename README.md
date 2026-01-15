# Face Blur Tool (YOLOv8 + Streamlit)
An AI-powered web application that automatically detects and blurs female faces in both images and videos. This project utilizes a custom-trained YOLOv8 model to ensure privacy while maintaining the integrity of the rest of the media.

# Features

Image Processing: Upload any JPG, PNG, or JPEG to instantly detect and blur female faces.

Video Processing: Supports MP4, AVI, and MOV formats with a real-time progress bar during processing.

Custom AI Model: Powered by a YOLOv8 model specifically trained for gender-based face detection.

Downloadable Results: Once processed, you can download the blurred images or videos directly to your device.

# Project Structure

app.py: The Streamlit-based web interface.

utils.py: Contains the core logic for image and video face-blurring.

train.py: The script used to train the YOLOv8 model.

requirements.txt: List of all necessary Python dependencies.

best.pt: Train Model.

# Installation
To run this project locally, follow these steps:

# Clone the repository:

Bash

git clone https://github.com/your-username/Face-Blur-App.git
cd Face-Blur-App
Install dependencies: Make sure you have Python installed, then run:

Bash

pip install -r requirements.txt
Run the Application:

Bash

streamlit run app.py

#  Requirements
The following libraries are required to run the app:

streamlit

ultralytics (YOLOv8)

opencv-python

numpy

torch

# How it Works
Detection: The app uses ultralytics (YOLO) to scan the media for faces.

Filtering: The system specifically checks for the class women (Class ID: 0).

Blurring: An adaptive Gaussian blur is applied to the detected regions to ensure the face is unrecognizable.
