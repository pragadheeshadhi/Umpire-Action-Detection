#pip install --upgrade --force-reinstall ultralytics
from ultralytics import YOLO
import torch
if __name__ == "__main__":
    # Initialize the model with a pre-defined configuration
    model = YOLO("yolov8n.pt")  # Usecva a model like yolov8n, yolov8s, etc.
    #model = YOLO("yolo11n.pt")  # Use a model like yolov8n, yolov8s, etc.
    # Train the model using your dataset configuration and specify the output directory
    device='cuda'
    model.train(
        data=r"c:\PROJECT\YOLO v8 datasets\only pitch.v2i.yolov8\data.yaml",  # Dataset configuration
        epochs=10,
        batch=16,
        imgsz=640,
        project=r"c:\PROJECT\YOLO v8 datasets\only pitch.v2i.yolov8",  # Path to save the training results
        name="Results10"  # Name of the folder under the project directory
    )
