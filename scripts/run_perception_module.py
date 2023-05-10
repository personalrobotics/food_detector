#!/usr/bin/python
import rospy
import os
import argparse

from visualization_msgs.msg import Marker

from pose_estimators.run_perception_module import run_detection
from pose_estimators.perception_module import PerceptionModule
from pose_estimators.marker_manager import MarkerManager

from food_detector.spnet_detector import SPNetDetector
from food_detector.retinanet_detector import RetinaNetDetector
from food_detector.spanet_detector import SPANetDetector

from food_detector.maskrcnn_detector import MaskRCNNDetector

import food_detector.ada_feeding_demo_config as conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Run perception module for ada feeding projects")
    parser.add_argument(
        "--demo-type", choices=['spnet', 'spanet', 'retinanet', 'maskrcnn'],
        required=True)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node('food_detector')

    rospy.init_node('food_detector')

    if args.demo_type == 'retinanet':
        pose_estimator = RetinaNetDetector(use_cuda=True, node_name=rospy.get_name())
    elif args.demo_type == "spnet":
        pose_estimator = SPNetDetector(use_cuda=conf.use_cuda, node_name=rospy.get_name())
    elif args.demo_type == "spanet":
        pose_estimator = SPANetDetector(use_cuda=conf.use_cuda)
    elif args.demo_type == "maskrcnn":
        pose_estimator = MaskRCNNDetector(use_cuda=conf.use_cuda)
    else:
        raise ValueError("Unknown demo type")

    if conf.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus
        print("Using CUDA!")

    marker_manager = MarkerManager(
        marker_type=Marker.CUBE,
        scale=[0.05, 0.01, 0.01],
        color=[0.5, 1.0, 0.5, 0.1],
        count_items=False)  # spnet and spanet handles this internally

    if args.demo_type != "maskrcnn":
        perception_module = PerceptionModule(
            pose_estimator=pose_estimator,
            marker_manager=marker_manager,
            detection_frame_marker_topic=None,
            detection_frame=conf.camera_tf,
            destination_frame=conf.destination_frame,
            purge_all_markers_per_update=True)

        run_detection(rospy.get_name(), conf.frequency, perception_module)
    else:
        perception_module = PerceptionModule(
            pose_estimator=pose_estimator,
            marker_manager=marker_manager,
            detection_frame_marker_topic=None,
            detection_frame=conf.maskrcnn_camera_tf,
            destination_frame=conf.maskrcnn_destination_frame,
            purge_all_markers_per_update=True,
            timeout=10
        )
        run_detection(rospy.get_name(), conf.maskrcnn_frequency, perception_module)

