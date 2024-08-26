from ultralytics import YOLO

#classification model
model = YOLO("/home/demir/Desktop/jhu_project/vein_puncture/runs/classify/mojtaba_test/weights/best.pt")

validation_results = model.val(data="/home/demir/Desktop/jhu_project/vein_puncture/puncture_detection_dataset", imgsz=640, split='test')


# object detection model
# model = YOLO("/home/demir/Desktop/vein-puncture-detection-egg/runs/detect/train7/weights/best.pt")

# validation_results = model.val(data="/home/demir/Desktop/vein-puncture-detection-egg/data/puncture_detection.v4i.yolov8/data.yaml", split='test')