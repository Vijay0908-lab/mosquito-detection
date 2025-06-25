from ultralytics import YOLO

model=YOLO('runs/detect/train6/weights/best.pt')

results=model.train(data='data.yaml',epochs=15,imgsz=128,device='cpu')
model.val()