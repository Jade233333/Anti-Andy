from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="config.yaml", epochs=1, device = "mps", resume=True, batch=16)

model = YOLO("yolov8n.yaml")
model.train(data="config.yaml", epochs=1, device = "mps", resume=True, batch=32)

model = YOLO("yolov8n.yaml")
model.train(data="config.yaml", epochs=1, device = "mps", resume=True, batch=64)