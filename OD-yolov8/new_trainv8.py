from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

model.train(data="config.yaml", epochs=100, device = "mps", resume=True, batch=32)