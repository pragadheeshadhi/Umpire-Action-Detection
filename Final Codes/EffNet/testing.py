import torch
import cv2
import timm
import numpy as np
from torchvision import transforms
from PIL import Image

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b0", pretrained=False)
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, 3)  # 3 classes: umpire, non-umpire, pitch
model.load_state_dict(torch.load("EfficientNet/Models/efficientnet_classifier_6.pth", map_location=device))
model.to(device)
model.eval()

# Define class labels
class_labels = ["non-umpire", "pitch","umpire"]
#class_labels = ["non-umpire","umpire"]
# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to classify a single frame
def classify_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return class_labels[predicted.item()]

# Process video
video_path = r"c:\Users\praga\Downloads\Untitled video - Made with Clipchamp (1).mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
output_video_path = "classified_video all6.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Classify frame
    label = classify_frame(frame)
    
    # Display result on the frame
    cv2.putText(frame,f"CLASS: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4)
    
    # Write frame to output video
    out.write(frame)
    
    # Show frame (optional)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved as {output_video_path}")
