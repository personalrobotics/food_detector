#!/usr/bin/python
# Example script for running a perception module

from pose_estimators.perception_module import PerceptionModule
from pose_estimators.marker_manager import MarkerManager
from pose_estimators.run_perception_module import run_detection
from food_detector_base.sim_food_detector import SimFoodDetector
import rospy


# This script runs a sim food detector which publishes markers for food items
# on the plate.

# When running without a robot, publish a static transform between map and another frame
# rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 map base_frame 1000
# You should be able to see marker array in rviz under topic /food_detector/marker_array
if __name__ == "__main__":
    detection_frame = "map"
    destination_frame = "map"

    rospy.init_node("food_detector")

    pose_estimator = SimFoodDetector(detection_frame)
    marker_manager = MarkerManager(count_items=False)

    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=marker_manager,
        detection_frame_marker_topic=None,
        detection_frame=detection_frame,
        destination_frame=destination_frame,
        purge_all_markers_per_update=True)

    destination_frame_marker_topic = rospy.get_name()
    frequency = 5
    run_detection(destination_frame_marker_topic, frequency, perception_module)
