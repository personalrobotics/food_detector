#!/usr/bin/env python

import rospy
import numpy as np

from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge

## probably need to create a new rosmsg, which contains:
## 1. Image 2. Push = push_vector/None. (None means that don't need to push)
# from food_detector.msg import AnnotatedImage

Repub = True

class RepubManager(object):
    """
    A class which act as an intermedia between camera image and SPANet.
    Subscribes to camera topics and  communicate with SPANet.
    """
    def __init__(self, node_name='RepubManager'):
        self.node_name = node_name
        self.image_topic='/camera/color/image_raw/compressed'
        self.spanet_in_topic = '/ReconfigManager/out/spanet'
        self.spanet_out_topic = '/ReconfigManager/in/spanet/marker_array'
        self.food_detector_topic = '/food_detector/marker_array'

        self.img_cnt = 0
        self.marker_array_cnt = 0

        self.topic_init()
        print("finish super init")
        
    def topic_init(self):
        self.img_sub = rospy.Subscriber(self.image_topic, CompressedImage,
                                               self.camera_callback, queue_size=1)

        self.spanet_sub = rospy.Subscriber(self.spanet_out_topic, MarkerArray,
                                               self.spanet_callback, queue_size=1)

        self.final_pub = rospy.Publisher(
            self.food_detector_topic,
            MarkerArray,
            queue_size=1)

        self.spanet_pub = rospy.Publisher(
            self.spanet_in_topic,
            CompressedImage,
            queue_size=1)

    def camera_callback(self, img):
        # print('get img message from camera', img.data)
        self.img_cnt += 1
        print('{}. get img message from camera'.format(str(self.img_cnt)))
        self.spanet_pub.publish(img)

    def spanet_callback(self, marker_array):
        self.marker_array_cnt += 1
        print('{}. get marker_array msg from spanet'.format(str(self.marker_array_cnt)))
        self.final_pub.publish(marker_array)

if __name__ == '__main__':
    rospy.init_node('RepubManager')
    rm = RepubManager()

    try: 
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    


