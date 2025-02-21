import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import os

# Load trained model
model_path = r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\Models\Cricket_LSTM_Model.h5"
model = tf.keras.models.load_model(model_path)

# List of umpire signals
#signals = [
#    "Out", "No-ball", "Wide ball", "Four runs", "Six Runs", "Byes",
#    "Dead Ball", "Short Run", "Penalty Runs", "Revoke Decision"
#]
signals = [    "Leg Byes"  ]
# Initialize MediaPipe Pose and Hands
mpPose = mp.solutions.pose
mpHands = mp.solutions.hands
pose = mpPose.Pose()
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

# Video input path
video_path = r"c:\PROJECT\Final Year Project\Umpire detection\out signals.mp4"
cap = cv2.VideoCapture(video_path)

# Variables for inference
label = "LOADING..."
n_time_steps = 10  # Same as used in training
lm_list = []

def extract_landmarks(results_pose, results_hands):
    """
    Extract pose and hand landmarks into a flattened list.
    """
    c_lm = []

    # Extract pose landmarks (Body)
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        c_lm.extend([0] * 132)  # 33 landmarks × 4 values each

    # Extract hand landmarks (Left & Right Hands)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                c_lm.extend([lm.x, lm.y, lm.z])
    else:
        c_lm.extend([0] * 126)  # 21 landmarks × 3 values each (for both hands)

    return c_lm

def detect(model, lm_list):
    """
    Perform inference using the trained model.
    """
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)  # Reshape for LSTM input
    results = model.predict(lm_list)
    predicted_class = np.argmax(results)
    label = signals[predicted_class]

def draw_class_on_image(label, img):
    """
    Display the predicted class on the image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img

# Warm-up frames before starting detection
warmup_frames = 0
i = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process pose and hands
    results_pose = pose.process(imgRGB)
    results_hands = hands.process(imgRGB)

    i += 1
    
    if i > warmup_frames:
        # Extract landmarks
        c_lm = extract_landmarks(results_pose, results_hands)
        lm_list.append(c_lm)

        # Perform inference when enough frames are collected
        if len(lm_list) == n_time_steps:
            threading.Thread(target=detect, args=(model, lm_list,)).start()
            lm_list = []  # Reset list for next detection cycle

        # Draw pose and hands on frame
        if results_pose.pose_landmarks:
            mpDraw.draw_landmarks(img, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Display the predicted class
    img = draw_class_on_image(label, img)

    cv2.imshow("Umpire Signal Detection", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
