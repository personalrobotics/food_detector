import sys
import rospy
import os
import json
import argparse

from visualization_msgs.msg import Marker

from deep_pose_estimators.run_perception_module import run_detection
from deep_pose_estimators.perception_module import PerceptionModule
from deep_pose_estimators.marker_manager import MarkerManager

import ada_feeding_demo_config as conf
from food_detector import FoodDetector


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run perception module for ada feeding projects")
    parser.add_argument("--demo-type", choices=['spnet-skewer', 'action-skewer'])
    args = parser.parse_args()

    rospy.init_node("food_detection")

    if args.demo_type == "spnet-skewer":
        import ada_feeding_demo_config as conf
        # TODO: shall we allow other options?
        pose_estimator = FoodDetector(use_spnet=True, use_cuda=True, use_model1=False)
    else:
        scores = json.loads("score.json")  # does not exists yet
        action_types = ["vertical_skewer_0",
                        "vertical_tilted_skewer_0",
                        "vertical_angled_skewer_0"]
        pose_estimator = ActionDetector(
            scores, action_types, conf.checkpoint,
            use_cuda=conf.use_cuda,
            label_map_file=conf.label_map,
            publisher_topic=conf.node_title)

    if conf.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    marker_manager = MarkerManager(marker_type=Marker.CUBE,
        scale=[0.01, 0.01, 0.01],
        color=[0.5, 1.0, 0.5, 0.1],
        count_items=True)

    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=marker_manager,
        detection_frame_marker_topic=None,
        detection_frame=conf.camera_tf,
        destination_frame="map",
        purge_all_markers_per_update=True)

    run_detection(conf.node_title, conf.frequency, perception_module)
