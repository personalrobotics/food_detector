#!/usr/bin/env python

import rospy
import numpy as np
from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem


class SimFoodDetector(PoseEstimator):
    def __init__(self, frame_id):

        # Pose at which the food is on the plate
        pose1 = np.array([[-1, 0, 0, 0.30],
                          [0, 1, 0, -0.25],
                          [0, 0, -1, 0.25],
                          [0, 0, 0, 1]])
        self.item1 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="cantaloupe",
            marker_id=1,
            db_key="food_item",
            pose=pose1,
            detected_time=rospy.Time.now())

        # Pose at which the food is on the plate
        pose2 = np.array([[-1, 0, 0, 0.25],
                          [0, 1, 0, -0.29],
                          [0, 0, -1, 0.25],
                          [0, 0, 0, 1]])
        self.item2 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="cantaloupe",
            marker_id=2,
            db_key="food_item",
            pose=pose2,
            detected_time=rospy.Time.now())

    def detect_objects(self):
        self.item1.detected_time = rospy.Time.now()
        self.item2.detected_time = rospy.Time.now()
        return [self.item1, self.item2]
