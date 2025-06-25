from ultralytics import YOLO

model=YOLO('9GB.pt')

results=model.train(data='data.yaml',epochs=10,imgsz=128,device='cpu')
model.val()
