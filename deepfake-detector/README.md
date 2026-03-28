# Real-Time Deepfake Detection System

A full-stack AI application that detects whether images, audio, or videos are real or AI-generated (deepfake). The system allows users to upload media and receive a prediction in real-time.

## Features

- **Image Deepfake Detection**: Uses a pretrained CNN (e.g., EfficientNet/ResNet) to classify an image as Real or AI-Generated.
- **Video Deepfake Detection**: Extracts frames from videos and analyzes them using a CNN to determine if the video is a deepfake.
- **Audio Deepfake Detection**: Converts audio to spectrograms and uses a deep learning model to detect synthetic voice patterns.
- **Real-Time Inference API**: A FastAPI/Flask backend serving the models.
- **Modern Web Interface**: A clean UI to upload media, view results, and see confidence scores.

## Project Structure

- `backend/`: Contains the training scripts and the `inference_api.py` server.
- `frontend/`: Contains the HTML, CSS, and JS files for the web interface.
- `models/`: Directory to store trained model files (`.pth` or `.h5`).
- `datasets/`: Directory to store datasets.

## Setup Instructions

### Backend Setup

1. Navigate to the `backend` directory.
2. Install the necessary requirements:
   ```bash
   pip install fastapi uvicorn torch torchvision torchaudio librosa opencv-python python-multipart pydantic
   ```
3. Run the API:
   ```bash
   uvicorn inference_api:app --reload
   ```

### Frontend Setup

1. Open `frontend/index.html` in your web browser.
2. Ensure the backend API is running locally to handle prediction requests.
