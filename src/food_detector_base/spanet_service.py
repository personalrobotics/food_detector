#!/usr/bin/python
import rospy
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from food_msg.srv import SPANetTrigger, SPANetTriggerResponse, SPANet, SPANetResponse
# from std_srvs.srv import Trigger, TriggerResponse

from retinanet_detector import RetinaNetDetector
from spanet_detector import SPANetDetector

from pose_estimators.perception_module import PerceptionModule
from pose_estimators.marker_manager import MarkerManager

import ada_feeding_demo_config as conf

br = CvBridge()
class SPANetService(object):

    def __init__(self, detector_type="RetinaNet"):

        self.service = rospy.Service('/SPANet', SPANet, self.handle_rqt)
        self.publisher = rospy.Publisher('/food_detector/marker_array', MarkerArray, queue_size=1)
        self.perception_init(detector_type)
        self.img_pub = rospy.Publisher('/food_detector/bbox_img', Image, queue_size=1)

    def perception_init(self, detector_type="RetinaNet"):
        # image_topic = '/sim_camera/color/image_raw/compressed'
        # depth_image_topic = '/sim_camera/aligned_depth_to_color/image_raw'
        image_topic = '/camera/color/image_raw/compressed'
        depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
        pose_estimator = None
        if detector_type == "RetinaNet":
            pose_estimator = RetinaNetDetector(use_cuda=True, node_name=rospy.get_name())
        if detector_type == "SPANet":
            pose_estimator = SPANetDetector(use_cuda=True, num_action_per_item=1, image_topic=image_topic, depth_image_topic=depth_image_topic)

        marker_manager = MarkerManager(
            marker_type=Marker.CUBE,
            scale=[0.05, 0.01, 0.01],
            color=[0.5, 1.0, 0.5, 0.1],
            count_items=False)  # spnet and spanet handles this internally

        self.perception_module = PerceptionModule(
            pose_estimator=pose_estimator,
            marker_manager=marker_manager,
            detection_frame_marker_topic=None,
            detection_frame=conf.camera_tf,
            destination_frame=conf.destination_frame,
            purge_all_markers_per_update=True)

    def convert(self, raw_img_msg):
        raw_img = br.imgmsg_to_cv2(raw_img_msg)
        print("get detect rqt from RM/pix2food")
        return raw_img

    def handle_rqt(self, rqt):
        raw_img = self.convert(rqt.raw_img_msg)
        detection_markers, bbox_img_msg = self.perception_module.get_detected_objects_as_markers(raw_img)
        print("finish one detection")
        return SPANetResponse(
            markers=detection_markers,
            bbox_img_msg=bbox_img_msg 
        )
