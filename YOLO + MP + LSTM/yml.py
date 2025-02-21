import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque

# 📌 Load YOLO Model (Umpire Detection)
yolo_model = YOLO(r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\YOLO + MP + LSTM\YOLO weights\300 Epoch\weights\best.pt")  # Your trained umpire detection model

# 📌 Load LSTM Model (Signal Classification)
lstm_model = load_model(r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\YOLO + MP + LSTM\LSTM weights\LSTM_Model_T7.h5")

# 📌 Define Class Labels
class_labels = ["No Action", "No Ball", "Four", "Wide", "Out", "Revoke", "Penalty", "Bye", "Leg Bye", "Short Run", "Six", "Dead Ball"]

# 📌 Define Video Path
video_path = r"c:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\Extended Cuts\four_extended.mp4"


# 📌 Initialize Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 📌 Function to extract landmarks
def extract_landmarks(results_pose, results_hands):
    pose_lm = [coord for lm in results_pose.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)] if results_pose.pose_landmarks else []
    hand_lm = [coord for hand in results_hands.multi_hand_landmarks for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)] if results_hands.multi_hand_landmarks else []
    
    # Ensure correct length (327)
    return pose_lm + [0.0] * (132 - len(pose_lm)) + hand_lm + [0.0] * (195 - len(hand_lm))

# 📌 Load video
#video_path = r"c:\PROJECT\Final Year Project\Umpire detection\umpire all signals.mp4"
cap = cv2.VideoCapture(video_path)

frame_buffer = deque(maxlen=72)  # Store 72 frames for LSTM prediction

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 📌 Detect umpire using YOLO
    results = yolo_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    if len(detections) == 0:
        cv2.putText(frame, "No Umpire Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue  # Skip this frame

    # 📌 Process only the first detected umpire
    x1, y1, x2, y2 = map(int, detections[0])  # Bounding box coordinates
    umpire_crop = frame[y1:y2, x1:x2]  # Crop to detected umpire

    # Convert to RGB
    umpire_rgb = cv2.cvtColor(umpire_crop, cv2.COLOR_BGR2RGB)

    # 📌 Process cropped region with Mediapipe
    results_pose, results_hands = pose.process(umpire_rgb), hands.process(umpire_rgb)
    frame_landmarks = extract_landmarks(results_pose, results_hands)

    frame_buffer.append(frame_landmarks)  # Store in buffer

    # 📌 Draw Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 📌 Draw Pose Landmarks
    if results_pose.pose_landmarks:
        mpDraw.draw_landmarks(frame, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    # 📌 Draw Hand Landmarks
    if results_hands.multi_hand_landmarks:
        for handLms in results_hands.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    # 📌 Make Prediction using LSTM Model
    if len(frame_buffer) == 72:
        prediction = lstm_model.predict(np.array(frame_buffer).reshape(1, 72, 327))
        label = class_labels[np.argmax(prediction)]
        cv2.putText(frame, f"Prediction: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()