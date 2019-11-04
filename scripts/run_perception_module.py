#!/usr/bin/python
import rospy
import os
import argparse

from visualization_msgs.msg import Marker

from pose_estimators.run_perception_module import run_detection
from pose_estimators.perception_module import PerceptionModule
from pose_estimators.marker_manager import MarkerManager

from food_detector.food_detector import FoodDetector
from food_detector.retinanet_detector import RetinaNetDetector
import food_detector.ada_feeding_demo_config as conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Run perception module for ada feeding projects")
    parser.add_argument(
        "--demo-type", choices=['spnet', 'action-score', 'retinanet'],
        required=True)
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("food_detector")

    if args.demo_type == 'retinanet':
        pose_estimator = RetinaNetDetector(use_cuda=True, node_name=rospy.get_name())
    elif args.demo_type == "spnet":
        rospy.init_node('food_detector')
        pose_estimator = FoodDetector(use_cuda=True, node_name=rospy.get_name())
    else:
        raise NotImplementedError

    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    marker_manager = MarkerManager(
        marker_type=Marker.CUBE,
        scale=[0.01, 0.01, 0.01],
        color=[0.5, 1.0, 0.5, 0.1],
        count_items=False)  # spnet handles this internally

    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=marker_manager,
        detection_frame_marker_topic=None,
        detection_frame=conf.camera_tf,
        destination_frame="map",
        purge_all_markers_per_update=True)

    run_detection(rospy.get_name(), conf.frequency, perception_module)
