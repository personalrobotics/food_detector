#!/usr/bin/env python

import rospy
import numpy as np
from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem

class SimFaceDetector(PoseEstimator):
    def __init__(self, frame_id):

        personPose = np.array([
            [0, 1, 0, 0.274968],
            [-1, 0, 0, 0.350025],
            [0, 0, 1, 0.752],
            [0, 0, 0, 1]])

        self.item = DetectedItem(
            frame_id=frame_id,
            marker_namespace="mouth",
            marker_id=1,
            db_key="mouth",
            pose=personPose,
            detected_time=rospy.Time.now(),
            info_map=dict(**{"mouth-status": "open"}))

    def detect_objects(self):
        self.item.detected_time = rospy.Time.now()
        return [self.item]
