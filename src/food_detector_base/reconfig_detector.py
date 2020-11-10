#!/usr/bin/env python

import numpy as np
import rospy
import cv2

from tf import TransformListener
from visualization_msgs.msg import Marker

from food_msg.srv import SPANet, SPANetRequest, Pix2Food, Pix2FoodRequest
from pytorch_pix2food.dataset.utils import generateActionImg
from pose_estimators.utils.ros_utils import get_transform_matrix
from pose_estimators.marker_manager import MarkerManager
from food_detector_base import ada_feeding_demo_config as conf
from food_detector_base import SPNetDetector, RetinaNetDetector, SPANetDetector
from food_detector_base import Visualizer

class ReconfigDetector(SPANetDetector, Visualizer):
    
    def __init__(self):
        SPANetDetector.__init__(self)
        Visualizer.__init__(self)
        self.init_pose_estimator_param()
        self.pix2food_client = rospy.ServiceProxy('pix2food', Pix2Food)

    def init_pose_estimator_param(self):
        self.listener = TransformListener()
        self.marker_manager = MarkerManager(
        marker_type=Marker.CUBE,
        scale=[0.05, 0.01, 0.01],
        color=[0.5, 1.0, 0.5, 0.1],
        count_items=False)  # spnet and spanet handles this internally

        self.destination_frame = 'map'
        self.detection_frame = conf.camera_tf

    def call_pix2food(self, startImg, actImg):
        rospy.wait_for_service("pix2food", timeout=5)
        startImg, actImg = self.br.cv2_to_imgmsg(startImg), self.br.cv2_to_imgmsg(actImg)
        res = self.pix2food_client(Pix2FoodRequest(
            startImg=startImg,
            actImg=actImg
        ))
        fakeImg = self.br.imgmsg_to_cv2(res.fakeImg)
        #TODO figure out the best API input data type? Check out pix2food model
        # print(np.max(fakeImg), np.min(fakeImg))
        cv2.normalize(fakeImg, fakeImg, 0, 255, cv2.NORM_MINMAX)
        fakeImg = fakeImg.astype(np.uint8)
        return fakeImg

    def generateActImg(self, bbox=None):
        if bbox is None:
            bbox=[75, 118, 300, 162]
            # bbox=[200, 138, 360, 192]
        else:
            forque_width = 45
            push_length = 225
            xmin, ymin, xmax, ymax = bbox
            ymean = (ymin + ymax) // 2
            Ymin = ymean - forque_width // 2
            Ymax = ymean + forque_width // 2 
            Xmax = xmax
            Xmin = Xmax - push_length
            bbox = [Xmin, Ymin, Xmax, Ymax]
        xmin, ymin, xmax, ymax = bbox
        start = [xmax, (ymin + ymax) // 2]
        end = [xmin, (ymin + ymax) // 2]
        actImg = generateActionImg(start, end, actImg=None, push_direction="left_push", 
                                img_size = (640, 480), forque_width=45)
        return actImg
    
    def get_detected_objects_as_markers(self, items):
        # Get the transform from destination to detection frame
        # print("self.destination_frame = ", self.destination_frame,  "self.detection_frame = ", self.detection_frame)
        markers = []
        frame_offset = get_transform_matrix(
            self.listener, self.destination_frame, self.detection_frame, timeout=2.0)
        # Convert items to be in desination frame
        for item in items:
            item.pose = np.dot(frame_offset, item.pose)
            item.frame_id = self.destination_frame
            markers += [self.marker_manager.item_to_marker(item)]

        return markers

    def detect_with_reconfig(self):

        # TODO change the RetinaNet API, return the bbox of each detected item
        raw_detections, raw_bbox_img_msg = self.detect_objects(self.img)
        raw_markers = self.get_detected_objects_as_markers(raw_detections)

        # generate pushed img
        bbox = None
        if len(raw_detections) > 0:
            bbox = raw_detections[0].bbox
        # print(bbox)
        self.left_action = self.generateActImg(bbox)
        self.left_img = self.call_pix2food(self.img, self.left_action)

        left_detections, left_bbox_img_msg = self.detect_objects(self.left_img)
        left_markers = self.get_detected_objects_as_markers(left_detections)
        
        left_bbox_img_with_actImg_msg = self.img_stiching(left_bbox_img_msg, self.left_action)
        return raw_markers, raw_bbox_img_msg, left_markers, left_bbox_img_with_actImg_msg

    def running(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            raw_detections, raw_bbox_img_msg, left_detections, left_bbox_img_with_actImg_msg = self.detect_with_reconfig()
            # TODO algorithm to find the best detections among pushed image & only publish the best one
            self.markers_pub.publish(raw_detections)
            self.raw_pub.publish(raw_bbox_img_msg)
            self.pushed_markers_pub.publish(left_detections)
            self.pushed_pub.publish(left_bbox_img_with_actImg_msg)
            rate.sleep()
        
if __name__ == '__main__':
    rospy.init_node("Reconfig_Detector")
    rd = ReconfigDetector()
    rd.running()
