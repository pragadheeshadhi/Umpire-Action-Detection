import cv2
import mediapipe as mp
import pandas as pd

# Read video from file
video_path = r"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\Extended Cuts\legbye_extended.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe Pose and Hands
mpPose = mp.solutions.pose
mpHands = mp.solutions.hands
pose = mpPose.Pose()
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "Leg Byes"
no_of_frames = 250

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

while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process pose and hands
    results_pose = pose.process(frameRGB)
    results_hands = hands.process(frameRGB)

    # Extract and store landmark data
    lm = extract_landmarks(results_pose, results_hands)
    lm_list.append(lm)

    # Draw Pose landmarks
    if results_pose.pose_landmarks:
        mpDraw.draw_landmarks(frame, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Draw Hand landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break



# Convert list to DataFrame
df = pd.DataFrame(lm_list)

# Define CSV file path
csv_file_path = rf"C:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\mp hand pose\Data\{label}.csv"

# Save to CSV instead of TXT
df.to_csv(csv_file_path, index=False, header=False)

cap.release()
cv2.destroyAllWindows()
print(f"Data saved successfully to {csv_file_path}")