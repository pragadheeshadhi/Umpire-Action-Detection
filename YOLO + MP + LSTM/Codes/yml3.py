import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time

#17903  kfb8S3  

# 📌 Load YOLO Model (Umpire Detection)
yolo_model = YOLO(r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\YOLO + MP + LSTM\YOLO weights\300 Epoch\weights\best.pt")  # Your trained umpire detection model

# 📌 Load LSTM Model (Signal Classification)
lstm_model = load_model(r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\Models4\LSTM_Model_T9.h5")

# 📌 Define Class Labels
class_labels = ["No Action", "No Ball", "Four", "Wide", "Out", "Revoke", "Penalty", "Bye", "Leg Bye", "Short Run", "Six", "Dead Ball"]

# 📌 Define Video Path
video_path = r"c:\Users\praga\Downloads\videos\four.mp4"

# 📌 Video Writer for saving output
output_path =  r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\YOLO + MP + LSTM\Outputs"

# 📌 Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 📌 Custom drawing styles
red_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
white_drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)

# 📌 Function to extract pose and hand landmarks
def extract_landmarks(results_pose, results_hands):
    pose_lm = [coord for lm in results_pose.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)] if results_pose.pose_landmarks else []
    hand_lm = [coord for hand in results_hands.multi_hand_landmarks for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)] if results_hands.multi_hand_landmarks else []
    return pose_lm + [0.0] * (132 - len(pose_lm)) + hand_lm + [0.0] * (195 - len(hand_lm))

# 📌 Function to transform landmarks to match original frame
def transform_landmarks(landmarks, x1, y1, x2, y2):
    transformed_landmarks = []
    for lm in landmarks:
        transformed_x = int(lm.x * (x2 - x1) + x1)
        transformed_y = int(lm.y * (y2 - y1) + y1)
        transformed_landmarks.append((transformed_x, transformed_y))
    return transformed_landmarks

# 📌 Load video
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 📌 Buffer for LSTM Predictions
frame_buffer = deque(maxlen=72)
time1=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 📌 Detect umpire using YOLO
    results = yolo_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()

    if len(detections) == 0:
        cv2.putText(frame, "No Umpire Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(frame)  # Save frame
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    # 📌 Process first detected umpire
    x1, y1, x2, y2 = map(int, detections[0])
    umpire_crop = frame[y1:y2, x1:x2]

    if umpire_crop.shape[0] < 50 or umpire_crop.shape[1] < 50:
        out.write(frame)
        cv2.imshow("Output", frame)
        continue

    # Convert to RGB for Mediapipe
    umpire_rgb = cv2.cvtColor(umpire_crop, cv2.COLOR_BGR2RGB)

    # 📌 Process cropped region with Mediapipe
    results_pose, results_hands = pose.process(umpire_rgb), hands.process(umpire_rgb)
    frame_landmarks = extract_landmarks(results_pose, results_hands)
    frame_buffer.append(frame_landmarks)

    # 📌 Draw Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 📌 Draw Pose Landmarks
    if results_pose.pose_landmarks:
        transformed_pose = transform_landmarks(results_pose.pose_landmarks.landmark, x1, y1, x2, y2)
        for lm in transformed_pose:
            cv2.circle(frame, lm, 3, (0, 0, 255), -1)  # Draw red dots for pose

    # 📌 Draw Hand Landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            transformed_hand = transform_landmarks(hand_landmarks.landmark, x1, y1, x2, y2)
            for lm in transformed_hand:
                cv2.circle(frame, lm, 3, (0, 0, 255), -1)  # Draw red dots for hands

    # 📌 Make Prediction using LSTM Model
    if len(frame_buffer) == 72:
        prediction = lstm_model.predict(np.array(frame_buffer).reshape(1, 72, 327))
        label = class_labels[np.argmax(prediction)]
        cv2.putText(frame, f"Prediction: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    time2 = time.time()
    if (time2 - time1) > 0:
        fps = 1.0 / (time2 - time1)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    time1 = time2
    
    # 📌 Save frame to video
    out.write(frame)
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release() 
cv2.destroyAllWindows()
print(f"Output video saved to {output_path}")