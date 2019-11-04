#!/usr/bin/env python

import rospy
import numpy as np
from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem


class SimFoodDetector(PoseEstimator):
    def __init__(self, frame_id):
        # middle of the plate
        pose0 = np.array([[1, 0, 0, 0.25],
                          [0, 1, 0, -0.29],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item0 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="cantaloupe",
            marker_id=0,
            db_key="food_item",
            pose=pose0,
            detected_time=rospy.Time.now())

        # 1st quadrant
        pose1 = np.array([[1, 0, 0, 0.25],
                          [0, 1, 0, -0.22],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item1 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="cantaloupe",
            marker_id=1,
            db_key="food_item",
            pose=pose1,
            detected_time=rospy.Time.now())

        # 2nd quadrant
        pose2 = np.array([[1, 0, 0, 0.18],
                          [0, 1, 0, -0.32],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item2 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="cantaloupe",
            marker_id=2,
            db_key="food_item",
            pose=pose2,
            detected_time=rospy.Time.now())

        # 3rd quadrant
        pose3 = np.array([[1, 0, 0, 0.25],
                          [0, 1, 0, -0.37],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item3 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="cantaloupe",
            marker_id=3,
            db_key="food_item",
            pose=pose3,
            detected_time=rospy.Time.now())

        # 4th quadrant
        pose4 = np.array([[1, 0, 0, 0.33],
                          [0, 1, 0, -0.29],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item4 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="cantaloupe",
            marker_id=4,
            db_key="food_item",
            pose=pose4,
            detected_time=rospy.Time.now())

    def detect_objects(self):
        self.item0.detected_time = rospy.Time.now()
        self.item1.detected_time = rospy.Time.now()
        self.item2.detected_time = rospy.Time.now()
        self.item3.detected_time = rospy.Time.now()
        self.item4.detected_time = rospy.Time.now()
        return [self.item2]
