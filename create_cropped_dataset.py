from ultralytics import YOLO
import os
import cv2

model = YOLO('path/to/run/folder/best.pt')

images_path = "data/Labeled_Images"

new_dataset_path = "data/cropped_images"

for label in os.listdir(images_path):
    for image in os.listdir(os.path.join(images_path, label)):
        result = model(os.path.join(images_path, label, image))

        x1, y1, x2, y2 = result[0].boxes.xyxyx.tolist()[0]
        cropped_image = cv2.imread(os.path.join(images_path, label, image))[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(new_dataset_path, label, image), cropped_image)

