import os
import cv2
from ultralytics import YOLO
import time
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from image_conversion_without_using_ros import image_to_numpy
import numpy as np
import torch

PUNCTURE_THRESHOLD = 0.8

needle_det_model = YOLO("crop_weights7.pt")

puncture_det_model = YOLO("classification_weights.pt")

def find_and_crop(image):
    start = time.perf_counter()
    result = needle_det_model(image)
    print(f"end: {time.perf_counter() - start:.4f}")
    try:
        x1, y1, x2, y2 = result[0].boxes.xyxy.tolist()[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(x1, y1, x2, y2)
    except IndexError:
        print("No needle detected")
        return image
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def detect_puncture(cropped_image):
    # Predict with the model
    results = puncture_det_model(cropped_image)  # predict on an image
    puncture_prob = results[0].probs.data.cpu().numpy()[2]
    print("PUNCTURE PROBABILITY:", puncture_prob)
    if puncture_prob >= PUNCTURE_THRESHOLD:
        puncture_flag = True
    else:
        puncture_flag = False
    return puncture_flag


# for image_file_name in os.listdir("path/to/images/folder"):
#     image = cv2.imread("path/to/images/folder/" + image_file_name)
#     cropped_image = find_and_crop(image)
#     puncture_flag = detect_puncture(cropped_image)
#     print(puncture_flag)


image_path = "/home/peiyao/Desktop/Mojtaba/vein-puncture-detection-egg/_recordings/Egg13_08_TP_bleeding/iOCT_image_448.3174460000.jpg"
image = cv2.imread(image_path)
cropped_image = find_and_crop(image)
puncture_flag = detect_puncture(cropped_image)
print(puncture_flag)