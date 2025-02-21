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
label = "Out"

# 📌 Create CSV File for the Label
csv_file = os.path.join(dataset_path, f"{label}.csv")
data_list = []

# 📌 Capture Video
cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera

if not cap.isOpened():
    print("❌ ERROR: Webcam not detected! Check camera index or permissions.")
    exit()

# 📌 Limit the number of frames
MAX_FRAMES = 500  # Change this to the desired number of frames
frame_count = 0

while frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Could not read frame. Exiting...")
        break

    # Convert to RGB for Mediapipe processing
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frameRGB)
    results_hands = hands.process(frameRGB)

    # 📌 Extract Pose Landmarks (Total: 132)
    pose_lm = []
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            pose_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
        mpDraw.draw_landmarks(frame, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
    else:
        pose_lm = [0.0] * 132  # Fill missing values

    # 📌 Extract Hand Landmarks (Total: 195)
    hand_lm = []
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                hand_lm.extend([lm.x, lm.y, lm.z])
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    
    # Ensure hand_lm has exactly 195 values
    if len(hand_lm) < 195:
        hand_lm.extend([0.0] * (195 - len(hand_lm)))

    # 📌 Combine Pose + Hand Landmarks (Total: 327)
    frame_landmarks = pose_lm + hand_lm

    # Ensure total size is 327 before storing
    if len(frame_landmarks) == 327:
        data_list.append(frame_landmarks)
        frame_count += 1  # Increment frame count
    else:
        print(f"❌ Incorrect landmark size: {len(frame_landmarks)}")

    # Display Label & Landmarks on Screen
    cv2.putText(frame, f"Recording: {label} ({frame_count}/{MAX_FRAMES})", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Recording Data", frame)

    # Press 'q' to stop recording early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 📌 Convert to DataFrame & Save CSV
df = pd.DataFrame(data_list, dtype=np.float32)
df.to_csv(csv_file, index=False, header=False)

print(f"✅ Data saved for label '{label}' at {csv_file} with {frame_count} frames")
