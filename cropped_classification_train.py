from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(data="/home/demir/Desktop/vein-puncture-detection-egg/data/cropped_images",  name="cropped_detection")

# TODO: change path to dataset folder!!