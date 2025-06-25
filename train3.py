from ultralytics import YOLO

model=YOLO('runs/detect/train7/weights/best.pt')

results=model.train(data='data.yaml',epochs=15,imgsz=256,device='cpu')
model.val()