import rospy
import os
import argparse

from visualization_msgs.msg import Marker

from deep_pose_estimators.run_perception_module import run_detection
from deep_pose_estimators.perception_module import PerceptionModule
from deep_pose_estimators.marker_manager import MarkerManager

import food_detector.ada_feeding_demo_config as conf
from food_detector import FoodDetector
from food_detector.retinanet_detector import RetinaNetDetector
from food_detector import ActionDetector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Run perception module for ada feeding projects")
    parser.add_argument(
        "--demo-type", choices=['spnet', 'spanet', 'retinanet'],
        required=True)
    args = parser.parse_args()

    if args.demo_type == 'retinanet':
        # TODO
        pose_estimator = RetinaNetDetector()
    elif args.demo_type == "spnet":
        import food_detector.ada_feeding_demo_config as conf
        # TODO: shall we allow other options?
        rospy.init_node(conf.node_name)
        pose_estimator = FoodDetector(use_cuda=True)
    elif args.demo_type == 'spanet':
        rospy.init_node(conf.node_name)
        pose_estimator = ActionDetector()

    if conf.use_cuda:
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

    print("run detection")
    run_detection(conf.node_name, conf.frequency, perception_module)
