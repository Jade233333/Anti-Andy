from ultralytics import YOLO

model = YOLO('/Users/jade/Developer/Anti-Andy/runs/detect/train10/weights/best.pt')

results = model.predict(source=['/Users/jade/Developer/Anti-Andy/test/3.jpg'], show = True, device='mps')

print(results)

x = input("c")