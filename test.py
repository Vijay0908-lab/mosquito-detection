from ultralytics import YOLO

model=YOLO('runs/detect/train2/weights/best.pt')

results=model.predict(source='images/test',stream=True,conf=0.3)

for (i,result) in enumerate(results):
    result.save(filename=f'{i}.jpg')