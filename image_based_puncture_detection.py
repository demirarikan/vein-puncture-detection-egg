import time
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from ultralytics import YOLO

rospy.init_node('image_puncture_detection', anonymous=True)
time.sleep(0.5)

# Load a model
model = YOLO(
    "/home/eyebrp/catkin_ws/vein-puncture-detection-egg/runs/classify/mojtaba_test/weights/best.pt"
)  # load a custom model

pub_puncture_flag = rospy.Publisher("PunctureFlagImage", Bool, queue_size=1)

def detect_puncture_camera(image):
    # Predict with the model
    results = model(image.data)  # predict on an image
    pred = results[0].probs
    if pred == 1:
        puncture_flag = True
    else:
        puncture_flag = False
    pub_puncture_flag.publish(puncture_flag)


iOCT_camera_sub = rospy.Subscriber(
    "/decklink/camera/image_raw", Image, detect_puncture_camera
)
# b_scan_sub = rospy.Subscriber("/b_scan", Image, detect_puncture_bscan)
