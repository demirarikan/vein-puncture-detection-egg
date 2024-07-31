import time
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ultralytics import YOLO
from image_conversion_without_using_ros import image_to_numpy

rospy.init_node('image_puncture_detection', anonymous=True)
time.sleep(0.5)

# Load a model
model = YOLO(
    "runs/classify/mojtaba_test/weights/best.pt"
)  # load a custom model

puncture_threshold = 0.8

pub_puncture_flag = rospy.Publisher("PunctureFlagImage", Bool, queue_size=1)

def detect_puncture_camera(image):
    iOCT_frame = image_to_numpy(image)
    iOCT_frame = cv2.resize(iOCT_frame, (640, 480))
    # Predict with the model
    results = model(iOCT_frame)  # predict on an image
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
