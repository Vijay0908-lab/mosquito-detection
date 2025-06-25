import supervision as sv
import numpy as np
from ultralytics import YOLO

Path='videos/mos0.mp4'

model=YOLO('runs/detect/train8/weights/best.pt')

video_info=sv.VideoInfo.from_video_path(Path)

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=256)[0]
    
    detections = sv.Detections.from_ultralytics(results)

    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator=sv.LabelAnnotator()

    labels = [f"{model.names[class_id]} {confidence:0.2f}" for confidence, class_id in zip(detections.confidence,detections.class_id)]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame,detections=detections,labels=labels)

    return frame

sv.process_video(source_path=Path, target_path=f"result.mp4", callback=process_frame)