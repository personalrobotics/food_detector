import numpy as np
import cv2
import pcl
import rospy
from sensor_msgs.msg import Image, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

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

        self.pub_table = rospy.Publisher(
            '{}/table_point_cloud2'.format(self.topic),
            pc2.PointCloud2,
            queue_size=2)
