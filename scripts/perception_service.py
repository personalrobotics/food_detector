#!/usr/bin/env python

import rospy 
import numpy as np

from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger

from pose_estimators.marker_manager import MarkerManager

from food_detector.srv import SPANetTrigger
from food_detector_base import SPANetService


if __name__ == "__main__":

    rospy.init_node('food_detector_server')
    print("hello")
    spanetService = SPANetService("SPANet")

    rospy.spin()


