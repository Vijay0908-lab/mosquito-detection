from ultralytics import YOLO

model=YOLO('C:/Users/Daksh Dixit/Desktop/mosquito_dataset/python 2/runs/detect/train8/weights/best.pt')

results=model.predict(source='C:/Users/Daksh Dixit/Desktop/mosquito_dataset/python 2/test/images',stream=True,save=True,plots=True)

for result in results:
    result.save()