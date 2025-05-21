import torch
import cv2
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from ultralytics import YOLO
import supervision as sv
import time

# Load EfficientNet Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b0", pretrained=False)
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 3)
model.load_state_dict(torch.load("EfficientNet/Models/efficientnet_classifier_4.pth", map_location=device))
model.to(device)
model.eval()

# Load LSTM Model
lstm_model = load_model("C:\PROJECT\MediaPipe\MP (hand + Pose) + LSTM\Models4\DataFinal_A1M1.h5")
yolo_model = YOLO("yolov8n.pt")
yolo_model.to('cpu')

# Define Class Labels
class_labels = ["non-umpire", "pitch", "umpire"]
#signal_labels = ["No Action", "Out", "Wide", "Four", "Six"]
signal_labels = ["No Action", "No Ball", "Four", "Wide", "Out",
                "Revoke", "Penality", "Bye", "Leg Bye",
                  "Short Run", "Six", "Dead Ball"]
#score_map = {"Four": 4, "Six": 6, "Out": "wicket", "Wide": 1, "No Ball": 1}
score_map = {"No Action":0, "No Ball":1, "Four":4, "Wide":1, "Out":"wicket",
                 "Revoke":1, "Penality":5, "Bye":0, "Leg Bye":0,
                  "Short Run":-1, "Six":6, "Dead Ball":0}
# Define polygonal zones
ZONE_COLORS = [(255, 0, 0), (0, 255, 0)]  # Blue & Green
ZONE_POINTS = [
    np.array([[767, 346], [768, 354], [1254, 356], [1251, 350]], dtype=np.int32),  # Blue Zone
    np.array([[592,608], [592, 619], [1379, 626], [1379, 613]], dtype=np.int32),  # Green Zone
]


# Define Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dictionary to track persons who entered a zone
tracked_persons = {}
# Dictionary to track flag status
zone_cross_flags = {}
# Dictionary to track return status
return_flags = {}
# Variable to track runs
runs = 0
wickets = 0
last_detected_signal = None
# Variable to track previous run increment status
run_incremented = False  
cooldown_frames = 15  # Number of frames to wait before allowing a new detection
cooldown_counter = 0  # Counter for cooldown

# Initialize Mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Function to classify a frame
def classify_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return class_labels[predicted.item()]

# Function to extract landmarks
def extract_landmarks(results_pose, results_hands):
    pose_lm = [coord for lm in results_pose.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)] if results_pose.pose_landmarks else []
    hand_lm = [coord for hand in results_hands.multi_hand_landmarks for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)] if results_hands.multi_hand_landmarks else []
    return pose_lm + [0.0] * (132 - len(pose_lm)) + hand_lm + [0.0] * (195 - len(hand_lm))

# Process video
video_path = r"Umpire Videos/Misc videos/5 signals trained.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Define output video writer
output_video_path = "y.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_buffer = deque(maxlen=10)
time1 = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))  # Resize frame
    label = classify_frame(frame)

    if label == "umpire":
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose, results_hands = pose.process(frameRGB), hands.process(frameRGB)
        frame_landmarks = extract_landmarks(results_pose, results_hands)
        frame_buffer.append(frame_landmarks)
        if results_pose.pose_landmarks:
            mpDraw.draw_landmarks(frame, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    # Draw Hand Landmarks
        if results_hands.multi_hand_landmarks:
            for handLms in results_hands.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        if len(frame_buffer) == 10:
            prediction = lstm_model.predict(np.array(frame_buffer).reshape(1, 10, 327))
            signal = signal_labels[np.argmax(prediction)]
            cv2.putText(frame, f"Signal: {signal}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            if cooldown_counter == 0:
                if signal in score_map:# and signal != last_detected_signal:
                    if score_map[signal] == "wicket":
                        wickets += 1
                    else:
                        runs += score_map[signal]
                    last_detected_signal = signal  # Store last detected signal
                    cooldown_counter = cooldown_frames  # Reset cooldown
        #cv2.putText(frame, f"Prediction: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 0, 0), 3)
        #cv2.putText(frame, f"Score: {runs}/{wickets}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    if label == "pitch":
         # Run YOLO with tracking
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.01, iou=0.1)  # Use ByteTrack
        if not results or results[0] is None:
            continue  # Skip if results are empty
        detections = sv.Detections.from_ultralytics(results[0])  # Access first result

        mask = detections.class_id == 0
        person_detections = detections[mask]

        # Check if tracker ID exists
        if person_detections.tracker_id is None:
            continue  # Skip frame if no tracking info

        # Draw zones
        for i, zone in enumerate(ZONE_POINTS):
            cv2.polylines(frame, [zone], isClosed=True, color=ZONE_COLORS[i], thickness=2)

        # Process detected persons
        for bbox, track_id in zip(person_detections.xyxy.astype(int), person_detections.tracker_id):
            x1, y1, x2, y2 = bbox
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            center_x, center_y = float(center_x), float(center_y)  # Ensure float type

            # If the person is already tracked, keep tracking them
            if track_id in tracked_persons:
                color = tracked_persons[track_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
                # Check if they crossed zones
                prev_zone = tracked_persons[track_id]
                for zone_index, zone in enumerate(ZONE_POINTS):
                    if cv2.pointPolygonTest(zone, (center_x, center_y), False) >= 0:
                        if prev_zone == ZONE_COLORS[0] and zone_index == 1:  # Blue -> Green
                            zone_cross_flags[track_id] = "Blue to Green"
                        elif prev_zone == ZONE_COLORS[1] and zone_index == 0:  # Green -> Blue
                            zone_cross_flags[track_id] = "Green to Blue"
                    
                        # Check if they returned to the original zone
                        if track_id in zone_cross_flags:
                            if (zone_cross_flags[track_id] == "Blue to Green" and zone_index == 0) or \
                            (zone_cross_flags[track_id] == "Green to Blue" and zone_index == 1):
                                return_flags[track_id] = "Returned to Original Zone"
                        break
                continue  # No need to check zones again

            # Check if the person entered any zone
            for zone_index, zone in enumerate(ZONE_POINTS):
                if cv2.pointPolygonTest(zone, (center_x, center_y), False) >= 0:
                    tracked_persons[track_id] = ZONE_COLORS[zone_index]  # Assign color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), ZONE_COLORS[zone_index], 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_COLORS[zone_index], 2)
                    break  # Stop checking zones

        # **Increment Run Logic**
        if len(zone_cross_flags) >= 2:
            runs += 1  # Increment run when an exchange happens
            zone_cross_flags.clear()  # Clear exchange tracking for next exchange

        if len(return_flags) >= 2:  
            runs += 1  # Increment run when players return
            return_flags.clear()  # Clear return tracking for next exchange


        # Display run count
        #cv2.putText(frame, f"Runs: {runs}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        # Display flags if a player crossed zones
        y_offset = 50
        for track_id, flag in zone_cross_flags.items():
            #cv2.putText(frame, f"{flag} - Player {track_id}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30

    # Display return flags
        for track_id, flag in return_flags.items():
            #cv2.putText(frame, f"{flag} - Player {track_id}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 30


    cv2.putText(frame, f"CLASS: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
    cv2.putText(frame, f"Score: {runs}/{wickets}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    out.write(frame)
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved as {output_video_path}")
