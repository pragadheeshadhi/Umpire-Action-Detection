import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# 📌 Load YOLO Model (Umpire Detection)
yolo_model = YOLO(r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\YOLO + MP + LSTM\YOLO weights\300 Epoch\weights\best.pt")  # Your trained umpire detection model

# 📌 Load LSTM Model (Signal Classification)
lstm_model = load_model(r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\YOLO + MP + LSTM\LSTM weights\LSTM_Model_T7.h5")

# 📌 Define Class Labels
class_labels = ["No Action", "No Ball", "Four", "Wide", "Out", "Revoke", "Penalty", "Bye", "Leg Bye", "Short Run", "Six", "Dead Ball"]

# 📌 Define Video Path
video_path = r"c:\Users\praga\Downloads\videos\four.mp4"

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
    
    return pose_lm + [0.0] * (132 - len(pose_lm)) + hand_lm + [0.0] * (195 - len(hand_lm))

# 📌 Load video
#video_path = r"c:\PROJECT\Final Year Project\Umpire detection\umpire all signals.mp4"
cap = cv2.VideoCapture(video_path)

# 📌 Video Writer for saving output
output_path = r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\YOLO + MP + LSTM\Outputs"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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
        out.write(frame)  # Save frame
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue  # Skip this frame

    # 📌 Process only the first detected umpire
    x1, y1, x2, y2 = map(int, detections[0])  # Bounding box coordinates
    umpire_crop = frame[y1:y2, x1:x2]  # Crop to detected umpire

    # Avoid processing too small detections
    if umpire_crop.shape[0] < 50 or umpire_crop.shape[1] < 50:
        out.write(frame)
        cv2.imshow("Output", frame)
        continue

    # Convert to RGB for Mediapipe
    umpire_rgb = cv2.cvtColor(umpire_crop, cv2.COLOR_BGR2RGB)

    # 📌 Process cropped region with Mediapipe
    results_pose, results_hands = pose.process(umpire_rgb), hands.process(umpire_rgb)
    frame_landmarks = extract_landmarks(results_pose, results_hands)

    frame_buffer.append(frame_landmarks)  # Store in buffer

    # 📌 Resize the landmarks back to original coordinates
    crop_h, crop_w, _ = umpire_crop.shape
    scale_x = (x2 - x1) / crop_w
    scale_y = (y2 - y1) / crop_h

    # 📌 Draw Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 📌 Draw Pose Landmarks in Correct Position
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            cx, cy = int(x1 + lm.x * crop_w * scale_x), int(y1 + lm.y * crop_h * scale_y)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # 📌 Draw Hand Landmarks in Correct Position
    if results_hands.multi_hand_landmarks:
        for handLms in results_hands.multi_hand_landmarks:
            for lm in handLms.landmark:
                cx, cy = int(x1 + lm.x * crop_w * scale_x), int(y1 + lm.y * crop_h * scale_y)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # 📌 Make Prediction using LSTM Model
    if len(frame_buffer) == 72:
        prediction = lstm_model.predict(np.array(frame_buffer).reshape(1, 72, 327))
        label = class_labels[np.argmax(prediction)]
        cv2.putText(frame, f"Prediction: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 📌 Save frame to video
    out.write(frame)
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
