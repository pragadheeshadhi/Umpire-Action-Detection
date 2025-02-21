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

# 📌 Dataset Storage Path
dataset_path = r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\data2"
os.makedirs(dataset_path, exist_ok=True)

# 📌 User Input: Label Name
label = "Zero"

# 📌 Create CSV File for the Label
csv_file = os.path.join(dataset_path, f"{label}.csv")
data_list = []

# 📌 Capture Video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Webcam not detected! Check camera index or permissions.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Could not read frame. Exiting...")
        break

    # Convert to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frameRGB)
    results_hands = hands.process(frameRGB)

    # 📌 Extract Pose Landmarks
    pose_lm = [0.0] * 132  # Default zero values
    if results_pose.pose_landmarks:
        temp_pose = []
        for lm in results_pose.pose_landmarks.landmark:
            temp_pose.extend([lm.x, lm.y, lm.z, lm.visibility])
        if len(temp_pose) == 132:
            pose_lm = temp_pose  # Use detected values

    # 📌 Extract Hand Landmarks
    hand_lm = [0.0] * 195  # Default zero values
    if results_hands.multi_hand_landmarks:
        temp_hand = []
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                temp_hand.extend([lm.x, lm.y, lm.z])
        if len(temp_hand) <= 195:
            hand_lm[:len(temp_hand)] = temp_hand  # Fill detected values

    # 📌 Combine Pose + Hands (Total: 327 values)
    frame_landmarks = pose_lm + hand_lm
    data_list.append(frame_landmarks)

    # Display Label & Landmarks on Screen
    cv2.putText(frame, f"Recording: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Recording Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 📌 Convert to DataFrame & Save
df = pd.DataFrame(data_list, dtype=np.float32)
df.to_csv(csv_file, index=False, header=False)

print(f"✅ Data saved for label '{label}' at {csv_file}")
