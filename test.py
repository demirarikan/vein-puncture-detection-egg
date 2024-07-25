from ultralytics import YOLO

model = YOLO("/home/demir/Desktop/jhu_project/vein_puncture/runs/classify/mojtaba_test/weights/best.pt")

validation_results = model.val(data="/home/demir/Desktop/jhu_project/vein_puncture/puncture_detection_dataset", imgsz=640, split='test')