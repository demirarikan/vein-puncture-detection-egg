from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(data="/home/demir/Desktop/vein-puncture-detection-egg/data/cropped_images", imgsz=640, device=[0,1], name="cropped_detection")