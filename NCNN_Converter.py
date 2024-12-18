from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model to NCNN format
model.export(format="ncnn", imgsz=640)  # creates 'yolo11n_ncnn_model'
