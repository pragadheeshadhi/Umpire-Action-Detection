from ultralytics import YOLO
import cv2

# Path to your trained model weights
model_path = r'YOLO weights/300 Epoch/weights/best.pt'

# Load the YOLOv8 model using the trained weights
model = YOLO(model_path)

# Path to the input video
input_video_path = r'c:\PROJECT\Umpire Environment\All umpire signals LSTM+Mediapipe\Extended Cuts\four_extended.mp4'  # Change to your video path
output_video_path = r'Outputs'  # Path to save the output video

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

# Create a VideoWriter to save the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run prediction on the current frame
    results = model.predict(frame, imgsz=640, conf=0.5)  # You can adjust confidence threshold if needed

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Write the frame to the output video
    out.write(annotated_frame)

    # Optional: Display the frame in a window (for debugging or preview)
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_video_path}")
