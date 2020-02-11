#!/usr/bin/env python

import rospy
import numpy as np

from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge
import cv2

img_path = '~/Dropbox/douwei.jpeg'
img_path = '/home/nansong/Dropbox/douwei.jpeg'

class TestRepubModule(object):

    def __init__(self, node_name='TestRepubModule'):
        self.image_topic='/camera/color/image_raw/compressed'
        self.spanet_in_topic = '/ReconfigManager/out/spanet'
        self.spanet_out_topic = '/ReconfigManager/in/spanet/marker_array'
        self.food_detector_topic = '/food_detector/marker_array'
        self.bridge = CvBridge()

        self.topic_init()

    def topic_init(self):

        self.final_sub = rospy.Subscriber(self.food_detector_topic, MarkerArray,
                                               self.final_callback, queue_size=1)

        self.spanet_sub = rospy.Subscriber(self.spanet_in_topic, CompressedImage,
                                               self.spanet_callback, queue_size=1)

        self.img_pub = rospy.Publisher(
            self.image_topic,
            CompressedImage,
            queue_size=1)

        self.spanet_pub = rospy.Publisher(
            self.spanet_out_topic,
            MarkerArray,
            queue_size=1)

    def spanet_callback(self, img):
        print('get img message from RepubManager')
        # print('get img message from camera:\n', img)
        ma = MarkerArray()
        ma.markers.append(Marker(text='test marker'))
        self.spanet_pub.publish(ma)

    def final_callback(self, marker_array):
        # print('get marker_array msg from spanet', marker_array.markers[0].text)
        print('get marker_array msg from spanet: it\'s ', marker_array.markers[0].text)

if __name__ == '__main__':
    node_name = 'TestRepubModule'
    rospy.init_node(node_name)
    trModule = TestRepubModule(node_name)

    img = cv2.imread(img_path)
    # msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
    # print("img: ", img)
    #### Create CompressedIamge ####    
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()
    cnt = 0
    try: 
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            cnt += 1
            print("send, {} time".format(str(cnt)))
            trModule.img_pub.publish(msg)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass