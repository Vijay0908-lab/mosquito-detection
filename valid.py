from ultralytics import YOLO

model=YOLO('runs/detect/train8/weights/best.pt')

model.predict('test/images',imgsz=256,plots=True,conf=0.3,save=True)