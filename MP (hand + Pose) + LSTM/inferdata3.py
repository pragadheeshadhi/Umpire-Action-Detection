import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# 📌 Load Model (Do NOT recompile)
model_path = r"Models1/LSTM_Model3.h5"
model = load_model(model_path)

# 📌 Define Class Labels
class_labels = ["Out", "Byes", "zero"]

# 📌 Initialize Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 📌 Extract Pose Landmarks
def extract_pose_landmarks(results_pose, frame):
    pose_lm = []
    if results_pose.pose_landmarks:
        mpDraw.draw_landmarks(frame, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for lm in results_pose.pose_landmarks.landmark:
            pose_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return pose_lm if len(pose_lm) == 132 else pose_lm + [0.0] * (132 - len(pose_lm))

# 📌 Extract Hand Landmarks
def extract_hand_landmarks(results_hands, frame):
    hand_lm = []
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                hand_lm.extend([lm.x, lm.y, lm.z])
    return hand_lm if len(hand_lm) == 194 else hand_lm + [0.0] * (194 - len(hand_lm))  # 🔥 Fix: 194 instead of 195

# 📌 Store Last 10 Frames (for LSTM)
frame_buffer = deque(maxlen=10)

# 📌 Start Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frameRGB)
    results_hands = hands.process(frameRGB)

    # Extract Landmarks & Draw Points
    pose_lm = extract_pose_landmarks(results_pose, frame)
    hand_lm = extract_hand_landmarks(results_hands, frame)

    # Combine Pose + Hands (Total: 326 values, NOT 327)
    frame_landmarks = pose_lm + hand_lm
    frame_buffer.append(frame_landmarks)  # Store in buffer

    # Only predict if we have 10 frames
    if len(frame_buffer) == 10:
        input_data = np.array(frame_buffer).reshape(1, 10, 326)  # 🔥 Fix: Match Model Shape (10, 326)

        # Make Prediction
        prediction = model.predict(input_data)
        predicted_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_index]

        # Display Prediction
        cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show Frame with Landmarks
    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
