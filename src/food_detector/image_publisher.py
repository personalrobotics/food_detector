import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImagePublisher(object):
    """
    A class which can subscribe to camera topics and publish detected images.
    """
    def __init__(self, node_name):
        self.node_name = node_name
        self.init_ros_publishers()

    def init_ros_publishers(self):
        self.pub_img = rospy.Publisher(
            '{}/detection_image'.format(self.node_name),
            Image,
            queue_size=2)
