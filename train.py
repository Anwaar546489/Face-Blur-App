from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # Check GPU
    print("CUDA available:", torch.cuda.is_available())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # or yolov8s.pt

    # Path to your dataset yaml
    data_yaml = r"C:\YOLO_FaceGender_Project\datasets\yolo_dataset\data.yaml"

    # Train model
    model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,        # use first GPU
        workers=3,
        name="face_gender_detector3",
        project=r"C:\YOLO_FACEGENDER_PROJECT\weights",
        exist_ok=True     # overwrite folder if it exists
    )