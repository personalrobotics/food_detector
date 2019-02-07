import sys
import rospy
import os
import ada_conf as conf
from deep_pose_estimators import run_detection

# TODO: may need to just remove it to reduce confusion.
# It is up to the user how to set this up (whether to read it from config file etc.)
if __name__ == '__main__':
    args = sys.argv

    config_filename = None
    if len(args) == 2:
        config_filename = args[1]
    else:
        ros_param_name = '/pose_estimator/config_filename'
        if rospy.has_param(ros_param_name):
            config_filename = rospy.get_param(ros_param_name)

    if config_filename is None:
        raise ValueError("Invalid arguments")

    os.environ['CUDA_VISIBLE_DEVICES'] = conf.gpus

    # TODO: setup the perception_module
    perception_module = None

    run_detection(conf.node_title, conf.frequency, perception_module)
