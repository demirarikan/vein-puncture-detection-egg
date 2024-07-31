#!/usr/bin/env python
import numpy as np
import cv2
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from pytransform3d.trajectories import plot_trajectory
import time
import math
# for ros stuff
import rospy
from std_msgs.msg import String, Float64MultiArray, Bool, Float64
from geometry_msgs.msg import Vector3, Transform
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import pandas as pd
# for dynamic reconfiguration of variables
import dynamic_reconfigure.client

# init variable
bridge = CvBridge()
isPunctureProbability = None
max_corr_values = None

vec3_x = None
vec3_y = None
vec3_z = None
rot_x  = None
rot_y = None
rot_z = None
rot_w = None
# usb_image = None
iOCT_image = None
b_scan = None

class ros_topics:

  def __init__(self):
    self.bridge = CvBridge()

    # subscribers
    self.transform_sub = rospy.Subscriber("/eye_robot/FrameEE", Transform, self.callback_2)
    # self.usb_camera_sub = rospy.Subscriber("/camera/image", Image, self.get_camera_image)
    self.iOCT_camera_sub = rospy.Subscriber("/decklink/camera/image_raw", Image, self.get_iOCT_camera_image)
    self.b_scan_sub = rospy.Subscriber("/b_scan", Image, self.get_b_scan_image)
    self.F_tip_sub = rospy.Subscriber("/eye_robot/TipForceNormGlobal", Float64, self.get_F_tip)
    self.F_tip_EMA_sub = rospy.Subscriber("/eye_robot/TipForceNormEMA_Global", Float64, self.get_F_tip_EMA)
    self.F_tip_AEWMA_sub = rospy.Subscriber("/eye_robot/TipForceNormAEWMA_Global", Float64, self.get_F_tip_AEWMA)
    self.F_tip_dot_sub = rospy.Subscriber("/eye_robot/TipForceNormDotGlobal", Float64, self.get_F_tip_dot)
    self.F_tip_dot_EMA_sub = rospy.Subscriber("/eye_robot/TipForceNormDotEMA_Global", Float64, self.get_F_tip_dot_EMA)
    self.F_tip_dot_AEWMA_sub = rospy.Subscriber("/eye_robot/TipForceNormDotAEWMA_Global", Float64, self.get_F_tip_dot_AEWMA)
    self.velocity_z_sub = rospy.Subscriber("/eye_robot/robotVelZ_Global", Float64, self.get_velocity_z)
    self.Puncture_detection_flag_sub = rospy.Subscriber("/eye_robot/PunctureDetectionFlag_Global", Int32, self.get_puncture_flag)

    

  def callback_2(self,data):
    global vec3_x
    global vec3_y
    global vec3_z
    global rot_x 
    global rot_y
    global rot_ziOCT_image
    vec3_z = data.translation.z
    rot_x = data.rotation.x
    rot_y = data.rotation.y
    rot_z = data.rotation.z
    rot_w = data.rotation.w

  # def get_camera_image(self,data):
  #   global usb_image
  #   usb_image = data
    
  def get_iOCT_camera_image(self,data):
    global iOCT_image
    iOCT_image = data

  def get_b_scan_image(self,data):
    global b_scan
    b_scan = data

  def get_F_tip(self, data): 
    global F_tip 
    F_tip = data.data

  def get_F_tip_EMA(self, data): 
    global F_tip_EMA 
    F_tip_EMA = data.data

  def get_F_tip_AEWMA(self, data): 
    global F_tip_AEWMA 
    F_tip_AEWMA = data.data

  def get_F_tip_dot(self, data): 
    global F_tip_dot 
    F_tip_dot = data.data 

  def get_F_tip_dot_EMA(self, data): 
    global F_tip_dot_EMA 
    F_tip_dot_EMA = data.data 

  def get_F_tip_dot_AEWMA(self, data): 
    global F_tip_dot_AEWMA 
    F_tip_dot_AEWMA = data.data 

  def get_puncture_flag(self, data):
    global puncture_flag 
    puncture_flag = data.data 

  def get_velocity_z(self, data):
    global velocity_z
    velocity_z = data.data 



# Create ROS publishers and subscribers
rt = ros_topics()
rospy.init_node('rostopic_recorder', anonymous=True)
time.sleep(0.5)
num_frames = 0
ee_points = []

# create a new dir
time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
ep_dir = os.path.join("_recordings", time_stamp)
os.makedirs(ep_dir)

rate = rospy.Rate(30) # ROS Rate at 5Hz

while not rospy.is_shutdown():    
    # Capture frame-by-frame
    print("No. ", num_frames)
    iOCT_frame = bridge.imgmsg_to_cv2(iOCT_image, desired_encoding = 'bgr8')
    iOCT_frame = cv2.resize(iOCT_frame, (640, 480))
    cv2.imshow('iOCT_frame', iOCT_frame)

    # save frame
    iOCT_save_name = os.path.join(ep_dir, "iOCT_image_{:06d}".format(num_frames) + ".jpg")
    # write the image
    cv2.imwrite(iOCT_save_name, iOCT_frame) # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)

    # save b_scan
    b_scan_frame = bridge.imgmsg_to_cv2(b_scan, desired_encoding = 'passthrough')
    b_scan_frame = cv2.resize(b_scan_frame, (640, 480))
    # cv2.imshow('b_scan_frame', b_scan_frame)

    # save frame
    b_scan_save_name = os.path.join(ep_dir, "b_scan_{:06d}".format(num_frames) + ".jpg")
    # write the image
    cv2.imwrite(b_scan_save_name, b_scan_frame) # UNCOMMENT ME WHEN DEBUGGING IS OVER (early)
    
    num_frames = num_frames + 1

    # get ee point
    ee_points.append([vec3_x, vec3_y, vec3_z,
                      rot_x, rot_y, rot_z, rot_w, velocity_z, F_tip, F_tip_EMA, F_tip_AEWMA, F_tip_dot, F_tip_dot_EMA, F_tip_dot_AEWMA, puncture_flag])

    if cv2.waitKey(1) & 0xFF == ord('q'):        
        break
    
    # make sure we spin at 30hz
    rate.sleep()


# save ee points
header =  ["vec3_x", "vec3_y", "vec3_z",
        "rot_x", "rot_y", "rot_z", "rot_w", "velocity_z", "F_tip", "F_tip_EMA", "F_tip_AEWMA", "F_tip_dot", "F_tip_dot_EMA", "F_tip_dot_AEWMA", "puncture_flag"]
csv_data = pd.DataFrame(ee_points)
ee_save_path = os.path.join(ep_dir, "ee_csv.csv")
csv_data.to_csv(ee_save_path, index = False, header = header)

# When everything done, destroy all windows
cv2.destroyAllWindows()
