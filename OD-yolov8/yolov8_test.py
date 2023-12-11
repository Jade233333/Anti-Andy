from ultralytics import YOLO

model = YOLO('/Users/jade/Developer/Anti-Andy/runs/detect/train/weights/best.pt')

results = model.predict(source=['/Users/jade/Developer/Anti-Andy/test/1.png'], show = True, device='cpu')

print(results)

x = input("c")