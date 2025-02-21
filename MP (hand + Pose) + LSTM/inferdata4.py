import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# 📌 Load Model
model = load_model("Models4/LSTM_Model_T2.h5")

# 📌 Define Class Labels
class_labels =  ["No Action", "No Ball","Four","Wide","Out","Bye","Six","Short Run","Leg Bye"] # Update based on training

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
    
    # Ensure correct length (327)
    return pose_lm + [0.0] * (132 - len(pose_lm)) + hand_lm + [0.0] * (195 - len(hand_lm))


# 📌 Start Webcam
cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=72)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose, results_hands = pose.process(frameRGB), hands.process(frameRGB)
    frame_landmarks = extract_landmarks(results_pose, results_hands)
    frame_buffer.append(frame_landmarks)

    if len(frame_buffer) == 72:
        prediction = model.predict(np.array(frame_buffer).reshape(1, 72, 327))
        label = class_labels[np.argmax(prediction)]
        cv2.putText(frame, f"Prediction: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Inference", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
