#!/usr/bin/env python

import rospy
import numpy as np
from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem


class SimFoodDetector(PoseEstimator):
    def __init__(self, frame_id):

        # # Pose at which the food is on the plate
        # pose1 = np.array([[1, 0, 0, 0.0],
        #                   [0, 1, 0, -0.40],
        #                   [0, 0, 1, 0.01],
        #                   [0, 0, 0, 1]])
        # self.item1 = DetectedItem(
        #     frame_id=frame_id,
        #     marker_namespace="cantaloupe",
        #     marker_id=1,
        #     db_key="food_item",
        #     pose=pose1,
        #     detected_time=rospy.Time.now(),
        #     info_map=dict(action="vertical", rotation=0.0, score=1.0, annotation='tv'))

        ## add additional info: push_direction, pushing_vec

        # Pose at which the food is on the plate
        pose0 = np.array([[1, 0, 0, 0.25],
                          [0, 1, 0, -0.29],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item0 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="carrot",
            marker_id=0,
            db_key="food_item",
            pose=pose0,
            detected_time=rospy.Time.now(),
            info_map=dict(action="vertical", rotation=0.0, score=1.0, annotation='tv', push_direction="left", pushing_vec=[0, 0, 1]))

        # 1th quadrant

        pose1 = np.array([[1, 0, 0, 0.25],
                          [0, 1, 0, -0.22],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item1 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="carrot",
            marker_id=1,
            db_key="food_item",
            pose=pose1,
            detected_time=rospy.Time.now(),
            info_map=dict(action="vertical", rotation=0.0, score=1.0, annotation='tv', push_direction="down", pushing_vec=[0, 0, 1]))

        # 4th quadrant
        pose4 = np.array([[1, 0, 0, 0.33],
                          [0, 1, 0, -0.29],
                          [0, 0, 1, 0.25],
                          [0, 0, 0, 1]])
        self.item4 = DetectedItem(
            frame_id=frame_id,
            marker_namespace="carrot",
            marker_id=4,
            db_key="food_item",
            pose=pose4,
            detected_time=rospy.Time.now(),
            info_map=dict(action="vertical", rotation=0.0, score=1.0, annotation='tv', push_direction="hi", pushing_vec=[0, 0, 1]))

    def detect_objects(self):
        self.item0.detected_time = rospy.Time.now()
        self.item1.detected_time = rospy.Time.now()
        # self.item2.detected_time = rospy.Time.now()
        # self.item3.detected_time = rospy.Time.now()
        self.item4.detected_time = rospy.Time.now()

        # return [self.item0] 
        # return [self.item1] 
        # return [self.item2] 
        # return [self.item3] 
        return [self.item4] 





