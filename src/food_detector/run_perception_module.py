import sys
import rospy
import os

import argparse
from deep_pose_estimators import run_detection
from deep_pose_estimators.perception_module import PerceptionModule
from food_detector import FoodDetector

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
    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=MarkerManager(),
        detection_frame_marker_topic=None,
        detection_frame=conf.camera_tf,
        destination_frame="map")

    # TODO: setup the perception_module

    run_detection(conf.node_title, conf.frequency, perception_module)
