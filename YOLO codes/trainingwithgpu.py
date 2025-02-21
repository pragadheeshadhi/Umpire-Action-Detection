from ultralytics import YOLO

if __name__ == '__main__':
    # Load YOLO model
    model = YOLO("yolov8n.pt")  # Choose the appropriate model size (n, s, m, etc.)

    # Train the model
    model.train(
        data=r"c:\PROJECT\YOLO v8 datasets\Umpire and Non-Umpire.v1i.yolov8\data.yaml",  # Path to updated YAML file
        epochs=5,      # Number of epochs
        imgsz=640,     # Image size
        project=r"c:\PROJECT\YOLO v8 datasets\Umpire and Non-Umpire.v1i.yolov8",  # Directory for saving runs
        name="Weights2",             # Name for this training session
        device='cpu'  # Use GPU for training
    )
