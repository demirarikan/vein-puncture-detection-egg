import time
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ultralytics import YOLO
from image_conversion_without_using_ros import image_to_numpy

rospy.init_node('image_puncture_detection', anonymous=True)
time.sleep(0.5)

# cropping model
needle_det_model = YOLO(
    "/home/demir/Desktop/vein-puncture-detection-egg/runs/detect/train7/weights/best.pt"  # TODO: change path to trained bounding box model
)

# detection classification model
puncture_det_model = YOLO(
    "/home/demir/Desktop/vein-puncture-detection-egg/runs/classify/cropped_detection6/weights/best.pt" # TODO: change path to trained classification model
) 

puncture_threshold = 0.8

pub_puncture_flag = rospy.Publisher("PunctureFlagImage", Bool, queue_size=1)
pub_cropped_image = rospy.Publisher("CroppedImage", Image, queue_size=1)

def find_and_crop(image):
    result = needle_det_model(image)

    try:
        x1, y1, x2, y2 = result[0].boxes.xyxy.tolist()[0]
    except IndexError:
        print("No needle detected")
        return image
    cropped_image = image[y1:y2, x1:x2]
    pub_cropped_image.publish(cropped_image)

    return cropped_image

def detect_puncture_camera(image):
    iOCT_frame = image_to_numpy(image)
    iOCT_frame = cv2.resize(iOCT_frame, (640, 480))

    # crop the needle
    cropped_image = find_and_crop(image)

    # Predict with the model
    results = puncture_det_model(cropped_image)  # predict on an image
    puncture_prob = results[0].probs.data.cpu().numpy()[2]
    print("punture prob:", puncture_prob)
    if puncture_prob >= puncture_threshold:
        puncture_flag = True
    else:
        puncture_flag = False
    pub_puncture_flag.publish(puncture_flag)


iOCT_camera_sub = rospy.Subscriber(
    "/decklink/camera/image_raw", Image, detect_puncture_camera
)
# b_scan_sub = rospy.Subscriber("/b_scan", Image, detect_puncture_bscan)

if __name__ == "__main__": 
    while not rospy.is_shutdown():
        pass 
