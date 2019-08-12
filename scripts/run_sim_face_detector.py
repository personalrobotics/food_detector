# Example script for running a perception module

from pose_estimators.perception_module import PerceptionModule
from pose_estimators.marker_manager import MarkerManager
from pose_estimators.run_perception_module import run_detection

from food_detector.sim_face_detector import SimFaceDetector
import rospy


# Run in command line a static transform between the detection frame to
# destination frame, and map to destination frame
# rosrun tf static_transform_publisher 0.0 0.0 0.0 0.0 0.0 0.0 1.0 map base_frame 1000
# You should be able to see marker array in rviz under topic /face_pose/marker_array

if __name__ == "__main__":
    detection_frame = "map"
    # Change to Robot Base Link, e.g.:
    destination_frame = "map"

    rospy.init_node("face_pose")

    pose_estimator = SimFaceDetector(detection_frame)
    marker_manager = MarkerManager(count_items=False)

    perception_module = PerceptionModule(
        pose_estimator=pose_estimator,
        marker_manager=marker_manager,
        detection_frame_marker_topic=None,  # Not used since pose estimator is provided.
        detection_frame=detection_frame,
        destination_frame=destination_frame,
        purge_all_markers_per_update=True)

    destination_frame_marker_topic = rospy.get_name()
    frequency = 5
    run_detection(destination_frame_marker_topic, frequency, perception_module)
