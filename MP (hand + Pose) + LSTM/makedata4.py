import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# 📌 Initialize Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 📌 Define Parameters
signal_name = "Dead Ball"  # Change signal name before recording
num_frames = 1000  # 📌 Limit on frames per recording
data = []

# 📌 Extract Pose Landmarks
def extract_pose_landmarks(results_pose):
    pose_lm = []
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            pose_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return pose_lm + [0.0] * (132 - len(pose_lm))  # Fill missing values

# 📌 Extract Hand Landmarks
def extract_hand_landmarks(results_hands):
    hand_lm = []
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                hand_lm.extend([lm.x, lm.y, lm.z])
    return hand_lm + [0.0] * (195 - len(hand_lm))  # Fill missing values

# 📌 Start Webcam
cap = cv2.VideoCapture(0)
frame_count = 0

while cap.isOpened() and frame_count < num_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frameRGB)
    results_hands = hands.process(frameRGB)

    # Extract Landmarks & Draw Points
    pose_lm = extract_pose_landmarks(results_pose)
    hand_lm = extract_hand_landmarks(results_hands)
    
    # Combine Pose + Hands (Total: 327 values)
    frame_landmarks = pose_lm + hand_lm
    frame_landmarks.append(signal_name)  # Append label

    data.append(frame_landmarks)  # Store Data
    frame_count += 1

    # Display Pose & Hand Landmarks
    if results_pose.pose_landmarks:
        mpDraw.draw_landmarks(frame, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Recording...", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 📌 Save Data to CSV
df = pd.DataFrame(data)
columns = [f"P{i}" for i in range(327)] + ["Label"]
df.columns = columns

# 📌 Create directory if not exists
save_path = "Data4"
os.makedirs(save_path, exist_ok=True)
df.to_csv(f"{save_path}/{signal_name}.csv", index=False)
print(f"Data saved for '{signal_name}' with {len(df)} frames.")
