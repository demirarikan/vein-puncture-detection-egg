from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(data="/home/demir/Desktop/jhu_project/vein_puncture/puncture_detection_dataset", epochs=55, imgsz=640, device=[0,1], val=False, name='mojtaba_test')