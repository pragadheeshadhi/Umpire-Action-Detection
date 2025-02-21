# Umpire Hand Signal Recognition

## Overview
This project implements real-time recognition of cricket umpire hand signals using **YOLOv8**, **OpenPose**, and **LSTM/GRU models**. It captures umpire signals from live video, classifies them, and updates the scorecard automatically.

## Features
- **Hand detection and keypoint extraction** using OpenPose.
- **YOLOv8-based object detection** for recognizing umpire signals.
- **LSTM/GRU-based classification** for accurate recognition of multiple cricket signals.
- **Real-time processing** with video input.
- **Dataset creation and training** for custom model improvement.

## Umpire Signals Covered
- Out
- No-ball
- Free-Hit
- Wide ball
- Four runs
- Six runs
- Byes
- Leg Byes
- Bouncer
- Television / 3rd Umpire
- Dead Ball
- Short Run
- Penalty Runs
- Revoke Decision
- Powerplay
- Soft Signal
- New Ball
- Last Hour

## Project Structure
```
├── data_collection
│   ├── collect_data.py  # Captures umpire hand keypoints and saves as CSV
│   ├── dataset/         # Stored CSV files with labeled umpire signals
│
├── training
│   ├── train_model.py   # Trains LSTM/GRU using dataset
│   ├── models/          # Saved trained models
│
├── inference
│   ├── test_model.py    # Runs inference on live video input
│   ├── demo_videos/     # Sample test videos
│
├── requirements.txt     # Dependencies for the project
├── README.md            # Project documentation
```

## Installation
### Prerequisites
- Python 3.8+
- OpenCV
- Mediapipe
- Ultralytics YOLOv8
- TensorFlow / PyTorch
- NumPy, Pandas, Matplotlib

### Still Work In Progress

## Results & Evaluation
- Accuracy and confusion matrix are displayed after training.
- Real-time predictions show detected umpire signals.

## Future Enhancements
- Improve dataset size for higher accuracy.
- Optimize YOLOv8 and OpenPose integration.
- Implement mobile deployment.

## Contributions
Feel free to contribute! Submit issues and pull requests.


