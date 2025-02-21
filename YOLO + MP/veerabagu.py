import math
import cv2
import mediapipe as mp
from ultralytics import YOLO
from time import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load YOLOv8 model
model = YOLO(r"c:\PROJECT\YOLO v8 datasets\Umpire-NonUmpire2.v1i.yolov8\Results200\weights\best.pt")

# Function to calculate angle between three points
def calculateAngle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

# Function to classify pose based on angle values
def classifyPose(landmarks, frame):
    label = 'Unknown Signal'
    color = (0, 0, 255)

    if not landmarks:
        return frame, label

    left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y))
    right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y))
    left_elbow = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x),
                  int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y))
    right_elbow = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x),
                   int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y))
    left_wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x),
                  int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y))
    right_wrist = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x),
                   int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y))

    left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y))
    right_knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x),
                  int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y))

    # Angle calculations for different body parts
    left_elbow_angle = calculateAngle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculateAngle(right_shoulder, right_elbow, right_wrist)
    left_shoulder_angle = calculateAngle(left_knee, left_shoulder, left_elbow)
    right_shoulder_angle = calculateAngle(right_knee, right_shoulder, right_elbow)
    left_knee_angle = calculateAngle(left_shoulder, left_knee, left_wrist)
    right_knee_angle = calculateAngle(right_shoulder, right_knee, right_wrist)

    print('Angles:', left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_knee_angle, right_knee_angle)

    # Define criteria for each umpire signal based on your angle ranges.
    if (left_elbow_angle > 90 and left_elbow_angle < 195 and 
        left_shoulder_angle > 0 and left_shoulder_angle < 30 and 
        right_shoulder_angle > 0 and right_shoulder_angle < 30 and 
        right_elbow_angle > 90 and right_elbow_angle < 195 and
        left_knee_angle > 150 and left_knee_angle < 195 and 
        right_knee_angle > 150 and right_knee_angle < 195):
        label = "No Signal"
        color = (255, 0, 0)
        
    # "Wide" Signal
    elif (left_elbow_angle > 100 and left_elbow_angle < 195 and 
          left_shoulder_angle > 40 and left_shoulder_angle < 100 and 
          right_shoulder_angle > 40 and right_shoulder_angle < 100 and 
          right_elbow_angle > 100 and right_elbow_angle < 195):
        label = "Wide Signal"
        color = (0, 255, 0)
    
    # "Six" Signal: Both arms raised straight above the head
    elif (left_elbow_angle > 100 and left_elbow_angle < 195 and 
          right_elbow_angle > 100 and right_elbow_angle < 195 and
          right_shoulder_angle > 130 and right_shoulder_angle < 195 and 
          left_shoulder_angle > 130 and left_shoulder_angle < 195):
        label = "Six Signal"
        color = (0, 255, 0)
    
    # "No Ball" Signal: Right arm extended horizontally at shoulder level.
    elif (right_elbow_angle > 100 and right_elbow_angle < 195 and 
          left_elbow_angle > 100 and left_elbow_angle < 195 and 
          right_shoulder_angle > 0 and right_shoulder_angle < 30 and 
          left_shoulder_angle < 90 and left_shoulder_angle > 0):
        label = "No Ball Signal"
        color = (0, 255, 0)

    # "Revoke" Signal: Both arms extended across at shoulder level.
    elif (right_elbow_angle > 40 and right_elbow_angle < 60 and 
          left_elbow_angle > 40 and left_elbow_angle < 60 and 
          right_shoulder_angle > 0 and right_shoulder_angle < 10 and 
          left_shoulder_angle > 0 and left_shoulder_angle < 10 and 
          left_knee_angle > 150 and left_knee_angle < 195 and 
          right_knee_angle > 150 and right_knee_angle < 195):
        label = "Revoke Signal"
        color = (0, 255, 0)
    
    # "Leg Bye" Signal: When knee angles indicate a leg position.
    elif ((left_knee_angle > 0 and left_knee_angle < 130) or 
          (right_knee_angle > 0 and right_knee_angle < 130)):
        label = "Leg Bye Signal"
        color = (0, 255, 0)

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return frame, label




# Code to test with an image
image_path = r"c:\Users\praga\Downloads\images11.jpeg"  # Replace with your image path

# Read the image
frame = cv2.imread(image_path)
frame = cv2.flip(frame, 1)

# Process the image using the YOLO model
results = model(frame)

# Loop over the results to find the umpire signal
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for i in range(len(boxes)):
        if class_ids[i] == 1 and confidences[i] > 0.5:  # Class "umpire"
            x1, y1, x2, y2 = map(int, boxes[i])

            umpire_region = frame[y1:y2, x1:x2]
            rgb_umpire_region = cv2.cvtColor(umpire_region, cv2.COLOR_BGR2RGB)

            results_pose = pose.process(rgb_umpire_region)

            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                for landmark in landmarks:
                    landmark.x = int(x1 + landmark.x * (x2 - x1))
                    landmark.y = int(y1 + landmark.y * (y2 - y1))

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                # Classify the pose and display the label
                frame, label = classifyPose(landmarks, frame)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the processed image
cv2.imshow("Umpire Pose Classification", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()





# Initialize video capture
video_path = r"c:\PROJECT\Final Year Project\Own dataset\six.mp4"
camera_video = cv2.VideoCapture(video_path)

time1 = 0

# Process video frame by frame
while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            if class_ids[i] == 1 and confidences[i] > 0.5:  # Class "umpire"
                x1, y1, x2, y2 = map(int, boxes[i])

                umpire_region = frame[y1:y2, x1:x2]
                rgb_umpire_region = cv2.cvtColor(umpire_region, cv2.COLOR_BGR2RGB)

                results_pose = pose.process(rgb_umpire_region)

                if results_pose.pose_landmarks:
                    landmarks = results_pose.pose_landmarks.landmark

                    for landmark in landmarks:
                        landmark.x = int(x1 + landmark.x * (x2 - x1))
                        landmark.y = int(y1 + landmark.y * (y2 - y1))

                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                    frame, label = classifyPose(landmarks, frame)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculate FPS
    time2 = time()
    if (time2 - time1) > 0:
        fps = 1.0 / (time2 - time1)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    time1 = time2

    cv2.imshow('Umpire Pose Classification', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera_video.release()
cv2.destroyAllWindows()