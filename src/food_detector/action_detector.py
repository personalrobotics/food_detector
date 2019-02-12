#!/usr/bin/env python

import json
import numpy as np
import rospy
from tf import TransformListener

from deep_pose_estimators.pose_estimators import PoseEstimator
from deep_pose_estimators.detected_item import DetectedItem
from deep_pose_estimators.utils.ros_utils import get_transform_matrix
from deep_pose_estimators.utils import CameraSubscriber

from food_detector import ImagePublisher, WallDetector, WallClass
from food_detector.util import load_retinanet, load_label_map

class ActionDetector(PoseEstimator, CameraSubscriber, ImagePublisher):
    """
    Action detector returns particular action as the class of each object.
    """

    def __init__(self, scores, action_types,
            retinanet_checkpoint,
            use_cuda,
            label_map_file,
            publisher_topic,
            image_topic='/camera/color/image_raw/compressed',
            image_msg_type='compressed',
            depth_image_topic='/camera/aligned_depth_to_color/image_raw',
            point_cloud_topic=None,
            camera_info_topic='/camera/color/camera_info',
            detection_frame = "camera_color_optical_frame",
            destination_frame = "map",
            timeout=1.0):
        """
        @param scores: food_class x pose (isolated vs wall) x action
                            -> success rate
        """
        PoseEstimator.__init__(self)
        CameraSubscriber.__init__(self, image_topic, image_msg_type,
            image_compressed, depth_image_topic, point_cloud_topic,
            camera_info_topic)
        ImagePublisher.__init__(self, publisher_topic)

        self.score = json.loads(scores)
        self.actions = self.action_types
        self.retinanet, self.retinanet_transform, self.encoder =
            load_retinanet(use_cuda, retinanet_checkpoint)
        self.label_map = load_label_map(label_map_file)

        self.listener = TransformListener()
        self.detection_frame = detection_frame
        self.destination_frame = destination_frame
        self.timeout

        # @Ethan This is the wall detector
        self.wall_detector = WallDetector()

    def detect_objects(self):
        if self.img_msg is None:
            print('no input stream')
            return list()

        if self.depth_img_msg is None:
            print('no input depth stream')
            self.depth_img_msg = np.ones(self.img_msg.shape[:2])

        # @Youngsun Get all bounding boxes and food classes
        detected_items = [] # List of (DetectedItem, image (u,v) of center of each item)

        # Get the transform from destination to detection frame
        camera_transform = get_transform_matrix(self.listener,
            self.destination_frame,
            self.detection_frame,
            self.timeout)

        # Register all UV Points in wall detector
        for _, uv in detected_items:
            self.wall_detector.register_uv(uv)

        for item, uv in detected_items:
            # @Ethan this is where the wall detector gets called
            wall_type = self.wall_detector.classify(uv, self.img_msg, self.depth_img_msg)

            scores = [self.score[item.namespace][wall_type][action]
                for action in self.actions]

            best_action = self.actions[np.argmax(scores)]

            item.info_map["best_action"] = best_action
            item.info_map["best_action_score"] = score

        return [item for item, uv in detected_items]