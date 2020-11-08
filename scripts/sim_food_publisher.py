#!/usr/bin/python

import rospy
import cv2
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, RegionOfInterest
from cv_bridge import CvBridge
br = CvBridge()

class SimFoodPublisher(object):

    def __init__(self,image_topic,depth_image_topic,camera_info_topic):
        self.image_topic = image_topic
        self.image_msg_type = CompressedImage if "compressed" in image_topic else Image
        self.depth_image_topic = depth_image_topic
        self.camera_info_topic = camera_info_topic

        self.img_pub = rospy.Publisher(self.image_topic, CompressedImage, queue_size=1)
        self.depth_img_pub = rospy.Publisher(self.depth_image_topic, Image, queue_size=1)
        self.info_pub = rospy.Publisher(self.camera_info_topic, CameraInfo, queue_size=1)

        print("Publish to ", self.image_topic)
        print("Publish to ", self.depth_image_topic)
        print("Publish to ", self.camera_info_topic)

    def start(self):
        # NOTE this script need to be run in the current path, since the image paths are relative path
        img_name="0061_5_finish.png"
        depth_name="0061_5_finish_depth.png"
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            finished = input("which pic to pub, 0 for raw, 1 for pushed")
            if finished == 0: self.load_img()
            if finished == 1: self.load_img(img_name, depth_name)
            self.img_pub.publish(self.compressed_img_msg)
            self.depth_img_pub.publish(self.depth_img)
            self.info_pub.publish(self.camInfo)
            r.sleep()

    def load_img(self, img_name='0060_1_start.png', depth_name='0060_1_start_depth.png'):
        cv_image = cv2.imread(img_name)
        self.compressed_img_msg = br.cv2_to_compressed_imgmsg(cv_image)
        depth_img = cv2.imread(depth_name, flags=cv2.CV_16UC1)
        self.depth_img = br.cv2_to_imgmsg(depth_img)

        self.camInfo = CameraInfo(
            header=Header(
                seq=0,
                stamp=rospy.Time.now(),
                frame_id="camera_color_optical_fram"
            ),
            height=480,
            width=640,
            distortion_model = "plumb_bob",
            D=[0.0, 0.0, 0.0, 0.0, 0.0],
            R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            K=[612.9367065429688, 0.0, 322.54913330078125, 0.0, 613.1729125976562, 248.15846252441406, 0.0, 0.0, 1.0],
            P=[612.9367065429688, 0.0, 322.54913330078125, 0.0, 0.0, 613.1729125976562, 248.15846252441406, 0.0, 0.0, 0.0, 1.0, 0.0],
            binning_x=0,
            binning_y=0,
            roi=RegionOfInterest(
                x_offset=0,
                y_offset=0,
                height=0,
                width=0,
                do_rectify=False
            )
        )

if __name__ == '__main__':
    rospy.init_node('image_publisher')
    image_topic = '/camera/color/image_raw/compressed'
    depth_image_topic = '/camera/aligned_depth_to_color/image_raw'    

    # image_topic = '/sim_camera/color/image_raw/compressed'
    # depth_image_topic = '/sim_camera/aligned_depth_to_color/image_raw'
    
    camera_info_topic = '/camera/color/camera_info'

    simFoodPub = SimFoodPublisher(image_topic, depth_image_topic, camera_info_topic)
    simFoodPub.start()

