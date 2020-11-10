import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker

class Visualizer(object):
    "viz module, edited by ed209"
    "test"
    def __init__(self):
        self.br = CvBridge()
        self.visual_pause = 0.1
        self.raw_pub = rospy.Publisher("/original/color/image", Image, queue_size=1)
        self.pushed_pub = rospy.Publisher("/pushed/color/image", Image, queue_size=1)
        self.markers_pub = rospy.Publisher("/food_detector/marker_array", MarkerArray, queue_size=1)
        self.pushed_markers_pub = rospy.Publisher("/pushed/marker_array", MarkerArray, queue_size=1)
        self.rate = rospy.Rate(20)

    def img_stiching(self, bbox_img_msg, actImg):
        bbox_img = self.br.imgmsg_to_cv2(bbox_img_msg)
        h, w = actImg.shape
        actImg_CH3 = np.zeros((h, w, 3), dtype=np.uint8)
        actImg_CH3[:,:,0] = actImg_CH3[:,:,1] = actImg_CH3[:,:,2] = actImg
        stiched_img = cv2.hconcat([bbox_img, actImg_CH3])
        return self.br.cv2_to_imgmsg(stiched_img, 'rgb8')

    def detectPlate(self, img, depth):
        # Detect Largest Circle (Plate)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self._hough_accum, self._hough_min_dist,
            param1=self._hough_param1, param2=self._hough_param2, minRadius=self._hough_min, maxRadius=self._hough_max)
        if circles is None:
            return None, None, None
        circles = np.round(circles[0, :]).astype("int")
        plate_uv = (0, 0)
        plate_r = 0
        for (x,y,r) in circles:
            print("Radius: " + str(r))
            if r > plate_r:
                plate_uv = (x, y)
                plate_r = r
        cv2.circle(img, plate_uv, plate_r + self._table_buffer, 1.0, -1)

    def getBBoxNearCenter(self, center, detections):
        min_dist = 100000
        best = detections[0].bbox
        for DetectItem in detections:
            xmin, ymin, xmax, ymax = DetectItem.bbox
            bbox_center = [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5]
            dist = np.linalg.norm(bbox_center - center)
            if dist < min_dist:
                best = DetectItem.bbox
                min_dist = dist
        
        return best
