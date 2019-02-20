import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImagePublisher(object):
    """
    A class which can subscribe to camera topics and publish detected images.
    """
    def __init__(self, topic):
        self.topic = topic
        self.init_ros_publishers()

    def init_ros_publishers(self):
        self.pub_img = rospy.Publisher(
            '{}/detection_image'.format(self.topic),
            Image,
            queue_size=2)
