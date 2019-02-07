import sys
import rospy
import os

import argparse
from deep_pose_estimators import run_detection
from deep_pose_estimators.perception_module import PerceptionModule
from food_detector import FoodDetector
from visualization_msgs.msg import Marker

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run perception module for ada feeding projects")
    parser.add_argument("--demo-type", choices=['spnet-skewer', 'qvalue-skewer'])
    args = parser.parse_args()

    rospy.init_node("food_detection")

    if args.demo_type == "spnet_skewer":
        import ada_feeding_demo_config as conf
        # TODO: shall we allow other options?
        pose_estimator = FoodDetector(use_spnet=True, use_cuda=True, use_model1=False)
    else:
        raise NotImplementedError

    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    marker_manager = MarkerManager(marker_type=Marker.CYLINDER,
        scale=[0.01, 0.01, 0.04],
        color=[0.5, 1.0, 0.5, 0.1],
        count_items=True)

    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=,
        detection_frame_marker_topic=None,
        detection_frame=conf.camera_tf,
        destination_frame="map",
        purge_all_markers_per_update=True)

    run_detection(conf.node_title, conf.frequency, perception_module)
