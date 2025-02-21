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
model = YOLO(r"c:\PROJECT\YOLO v8 datasets\Umpire and Non-Umpire.v1i.yolov8\Weights2\weights\best.pt")  # Replace with your YOLOv8 model path

# Function to calculate angle between three points
def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    x3, y3 = landmark3.x, landmark3.y

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

# Function to classify pose based on angles
def classifyPose(landmarks, output_image):
    label = 'Unknown Signal'
    color = (0, 0, 255)  

    # Calculate angles
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Classify signals based on angles
    if (left_elbow_angle > 100 and left_elbow_angle < 195 and 
        right_elbow_angle > 100 and right_elbow_angle < 195 and
        right_shoulder_angle > 130 and right_shoulder_angle < 195 and left_shoulder_angle > 130 and left_shoulder_angle < 195):
        label = "Six Signal"
        color = (0, 255, 0)
    
    elif (left_elbow_angle > 100 and left_elbow_angle < 195 and 
          left_shoulder_angle > 40 and left_shoulder_angle < 100 and 
          right_shoulder_angle > 40 and right_shoulder_angle < 100 and 
          right_elbow_angle > 100 and right_elbow_angle < 195):
        label = "Wide Signal"
        color = (255, 0, 0)

    elif (right_elbow_angle > 40 and right_elbow_angle < 60 and 
          left_elbow_angle > 40 and left_elbow_angle < 60 and 
          right_shoulder_angle > 0 and right_shoulder_angle < 10 and left_shoulder_angle > 0 and left_shoulder_angle < 10):
        label = "Revoke Signal"
        color = (0, 255, 255)

    # Draw label
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return output_image, label

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
                    # Adjust landmarks to fit in full-frame
                    for landmark in results_pose.pose_landmarks.landmark:
                        landmark.x = (landmark.x * (x2 - x1)) + x1
                        landmark.y = (landmark.y * (y2 - y1)) + y1
                    
                    # Draw corrected landmarks
                    mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Classify pose
                    frame, label = classifyPose(results_pose.pose_landmarks.landmark, frame)

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
