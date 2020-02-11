#!/usr/bin/env python

import rospy
import numpy as np

from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import yaml

from TestRepubModule import TestRepubModule

img_path = '~/Dropbox/douwei.jpeg'
img_path = '/home/nansong/Dropbox/douwei.jpeg'

class TestReconfigModule(TestRepubModule):
    def __init__(self, node_name='TestReconfigModule'):
        TestRepubModule.__init__(self, node_name=node_name)

    def spanet_callback(self, img):
        # self.count = (self.count + 1) % 2
        # print('get img message from ReconfigManager')
        yaml_node = yaml.load(img.header.frame_id)

        # print(yaml_node['push_direction'])

        if yaml_node['push_direction'] is None:
            scores = [0.7, 0.4]
        else:
            scores = self.fake_scores(yaml_node['push_direction'])
        
        ma = MarkerArray()
        info_map = dict(
            actions=['scoop-side', 'scoop-foward'],
            scores=scores,
            push_direction=yaml_node['push_direction'],   ## default to be 'left' 
            push_vec=yaml_node['push_vec'])   ## default to be along x axis. 
        mark = Marker(header=img.header, text=yaml.dump(info_map))
        # print(img.header)
        ma.markers.append(mark)

        self.spanet_pub.publish(ma)

    def final_callback(self, marker_array):
        # print('get marker_array msg from spanet', marker_array.markers[0].text)
        yaml_node = yaml.load(marker_array.markers[0].text)
        print('get marker_array msg from spanet: it\'s ', marker_array)
        
    def fake_scores(self, push_direction):
        scores_dict = {'left':[0.9, 0.5], 'up':[0.6, 0.4], 'right':[0.5, 0.1], 'down':[0.8, 0.1]}
        return scores_dict[push_direction]


if __name__ == '__main__':
    node_name = 'TestRepubModule'
    rospy.init_node(node_name)
    trModule = TestReconfigModule(node_name)

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
        while not rospy.is_shutdown() and cnt < 3:
            print("send, {} time".format(str(cnt)))
            msg.header.seq = cnt
            trModule.img_pub.publish(msg)
            rate.sleep()
            cnt += 1

    except rospy.ROSInterruptException:
        pass