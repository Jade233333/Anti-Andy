from ultralytics import YOLO

model = YOLO('/Users/jade/Developer/Anti-Andy/runs/detect/train10/weights/last.pt')

model.train(data="config.yaml", epochs=100, device = "mps", resume=True, batch=32)