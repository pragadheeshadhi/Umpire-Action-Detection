import math
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from time import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load YOLOv8 model
model = YOLO(r"c:\PROJECT\YOLO v8 datasets\Umpire and Non-Umpire.v1i.yolov8\Weights2\weights\best.pt")  # Replace with your model path

# Function to calculate angle between three points
def calculateAngle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

# Function to classify pose based on angles
def classifyPose(landmarks, frame):
    label = 'Unknown Signal'
    color = (0, 0, 255)

    if not landmarks:
        return frame, label

    # Extract required landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # Convert normalized coordinates to pixel values
    left_shoulder = (int(left_shoulder.x), int(left_shoulder.y))
    right_shoulder = (int(right_shoulder.x), int(right_shoulder.y))
    left_elbow = (int(left_elbow.x), int(left_elbow.y))
    right_elbow = (int(right_elbow.x), int(right_elbow.y))
    left_wrist = (int(left_wrist.x), int(left_wrist.y))
    right_wrist = (int(right_wrist.x), int(right_wrist.y))

    # Calculate angles
    left_elbow_angle = calculateAngle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculateAngle(right_shoulder, right_elbow, right_wrist)

    # Classify signals
    if 100 < left_elbow_angle < 195 and 100 < right_elbow_angle < 195:
        label = "Six Signal"
        color = (0, 255, 0)

    elif 40 < left_elbow_angle < 100 and 40 < right_elbow_angle < 100:
        label = "Wide Signal"
        color = (255, 0, 0)

    elif 40 < right_elbow_angle < 60 and 40 < left_elbow_angle < 60:
        label = "Revoke Signal"
        color = (0, 255, 255)

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return frame, label

# Initialize video capture
video_path = r"c:\PROJECT\Final Year Project\Own dataset\six.mp4"
camera_video = cv2.VideoCapture(video_path)

time1 = 0

# Process video frame by frame
while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    results = model(frame)  # Run YOLOv8 detection

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            if class_ids[i] == 1 and confidences[i] > 0.5:  # Class "umpire"
                x1, y1, x2, y2 = map(int, boxes[i])  # Convert coordinates to integers

                # Crop umpire region
                umpire_region = frame[y1:y2, x1:x2]
                rgb_umpire_region = cv2.cvtColor(umpire_region, cv2.COLOR_BGR2RGB)

                # Pose detection
                results_pose = pose.process(rgb_umpire_region)

                if results_pose.pose_landmarks:
                    landmarks = results_pose.pose_landmarks.landmark

                    # Adjust landmarks from cropped region back to full frame
                    for landmark in landmarks:
                        landmark.x = x1 + landmark.x * (x2 - x1)
                        landmark.y = y1 + landmark.y * (y2 - y1)

                    # Convert landmarks to pixel format
                    landmark_points = [(int(l.x), int(l.y)) for l in landmarks]

                    # Draw landmarks on frame
                    for pt in landmark_points:
                        cv2.circle(frame, pt, 5, (255, 0, 0), -1)

                    # Draw pose connections
                    mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Classify pose
                    frame, label = classifyPose(landmarks, frame)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculate FPS
    time2 = time()
    if (time2 - time1) > 0:
        fps = 1.0 / (time2 - time1)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    time1 = time2

    cv2.imshow('Umpire Pose Classification', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera_video.release()
cv2.destroyAllWindows()
