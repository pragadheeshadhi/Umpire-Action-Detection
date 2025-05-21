import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time

# 📌 Load Model
model = load_model(r"C:\PROJECT\MediaPipe\MP (hand + Pose) + LSTM\Models4\Data5_A1M1.h5")

# 📌 Define Class Labels and Score Mapping
#class_labels = ["No Action", "Out", "Wide", "Four", "Six"]
class_labels = ["No Action", "No Ball", "Four", "Wide", "Out",
                "Revoke", "Penality", "Bye", "Leg Bye",
                  "Short Run", "Six", "Dead Ball"]
score_map = {"No Action":0, "No Ball":1, "Four":4, "Wide":1, "Out":"wicket",
                 "Revoke":1, "Penality":5, "Bye":0, "Leg Bye":0,
                  "Short Run":-1, "Six":6, "Dead Ball":0}

# 📌 Initialize Score Variables
runs = 0
wickets = 0
last_detected_signal = None  # Stores the last detected signal
cooldown_frames = 15  # Number of frames to wait before allowing a new detection
cooldown_counter = 0  # Counter for cooldown

# 📌 Initialize Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 📌 Function to Extract Landmarks
def extract_landmarks(results_pose, results_hands):
    pose_lm = [coord for lm in results_pose.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)] if results_pose.pose_landmarks else []
    hand_lm = [coord for hand in results_hands.multi_hand_landmarks for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)] if results_hands.multi_hand_landmarks else []
    return pose_lm + [0.0] * (132 - len(pose_lm)) + hand_lm + [0.0] * (195 - len(hand_lm))  # Ensure length 327

# 📌 Start Video
video_path = r"c:\Users\praga\Downloads\all signals.mp4"
cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=10)
output_video_path = "Score Updationx.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, fourcc, fps, ( int(cap.get(3)),int(cap.get(4))))
time1 = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose, results_hands = pose.process(frameRGB), hands.process(frameRGB)
    frame_landmarks = extract_landmarks(results_pose, results_hands)
    frame_buffer.append(frame_landmarks)

    # Draw Landmarks
    if results_pose.pose_landmarks:
        mpDraw.draw_landmarks(frame, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
    if results_hands.multi_hand_landmarks:
        for handLms in results_hands.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    # Predict Signal & Update Score
    if len(frame_buffer) == 10:
        prediction = model.predict(np.array(frame_buffer).reshape(1, 10, 327))
        label = class_labels[np.argmax(prediction)]

        # Cooldown logic to prevent duplicate detections
        if cooldown_counter == 0:
            if label in score_map and label != last_detected_signal:
                if score_map[label] == "wicket":
                    wickets += 1
                else:
                    runs += score_map[label]
                last_detected_signal = label  # Store last detected signal
                cooldown_counter = cooldown_frames  # Reset cooldown

        # Display Prediction & Score
        cv2.putText(frame, f"Prediction: {label}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Score: {runs}/{wickets}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Reduce cooldown counter
    if cooldown_counter > 0:
        cooldown_counter -= 1

    # Calculate FPS
    time2 = time.time()
    if (time2 - time1) > 0:
        fps = 1.0 / (time2 - time1)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    time1 = time2
    out.write(frame)
    cv2.imshow("Inference", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
