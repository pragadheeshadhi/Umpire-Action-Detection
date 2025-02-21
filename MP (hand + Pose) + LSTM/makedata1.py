import cv2
import mediapipe as mp
import pandas as pd
import csv
import os

# 📌 Initialize Mediapipe for Pose & Hands
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 📌 Define Save Directory
save_dir = r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\Data1"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn’t exist

# 📌 Load Video
video_path = r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\Extended Cuts\legbye_extended.mp4"
cap = cv2.VideoCapture(0)

# 📌 Parameters
label = "Zero"  # Change label for different signals
no_of_frames = 250
lm_list = []

# 📌 Extract Pose Landmarks
def extract_pose_landmarks(results_pose):
    pose_lm = []
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            pose_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return pose_lm if len(pose_lm) == 63 else pose_lm + [0.0] * (63 - len(pose_lm))

# 📌 Extract Hand Landmarks
def extract_hand_landmarks(results_hands):
    hand_lm = []
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                hand_lm.extend([lm.x, lm.y, lm.z])
    return hand_lm if len(hand_lm) == 195 else hand_lm + [0.0] * (195 - len(hand_lm))

# 📌 Save Data to CSV
def save_to_csv(filename, data, label):
    cleaned_data = [float(x) if x != "" else 0.0 for x in data]  # Fill empty spaces with 0.0
    file_path = os.path.join(save_dir, filename)  # Ensure correct directory
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cleaned_data + [label])  # Append label at the end

# 📌 Main Data Collection Loop
while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frameRGB)
    results_hands = hands.process(frameRGB)

    # Extract Landmarks
    pose_lm = extract_pose_landmarks(results_pose)
    hand_lm = extract_hand_landmarks(results_hands)

    # Combine Pose + Hands (Total: 258 values)
    frame_landmarks = pose_lm + hand_lm

    # Append to List & Save
    lm_list.append(frame_landmarks)
    save_to_csv(f"{label}.csv", frame_landmarks, label)

    # Display Frame
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
