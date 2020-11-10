#!/usr/bin/env python

import numpy as np
import rospy
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
import torch
import yaml 
import cv2

from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem
from pose_estimators.utils import CameraSubscriber
from tf.transformations import quaternion_matrix, quaternion_from_euler

from image_publisher import ImagePublisher
from util import load_retinanet, load_label_map
import ada_feeding_demo_config as conf


class RetinaNetDetector(PoseEstimator, CameraSubscriber, ImagePublisher):
    def __init__(
            self,
            use_cuda,
            node_name=rospy.get_name(),
            retinanet_checkpoint=conf.checkpoint,
            label_map_file=conf.label_map,
            camera_tilt=1e-5,
            camera_to_table=conf.camera_to_table,
            frame=conf.camera_tf,
            image_topic='/camera/color/image_raw/compressed',
            image_msg_type='compressed',
            depth_image_topic='/camera/aligned_depth_to_color/image_raw',
            camera_info_topic='/camera/color/camera_info',
            detection_frame='camera_color_optical_frame',
            timeout=1.0,
            ema_alpha=1.0):

        PoseEstimator.__init__(self)
        CameraSubscriber.__init__(
            self,
            image_topic=image_topic,
            image_msg_type=image_msg_type,
            depth_image_topic=depth_image_topic,
            pointcloud_topic=None,
            camera_info_topic=camera_info_topic)
        ImagePublisher.__init__(self, node_name)

        self.retinanet, self.retinanet_transform, self.encoder = \
            load_retinanet(use_cuda, retinanet_checkpoint)
        self.label_map = load_label_map(label_map_file)

        self.timeout = timeout

        self.use_cuda = use_cuda

        self.camera_tilt = camera_tilt
        self.camera_to_table = camera_to_table
        self.frame = frame
        self.ema_alpha = ema_alpha

        self.detected_item_boxes = dict()

        self.z = None

    def create_detected_item(self, rvec, tvec, t_class_name, box_id, bbox,
                             db_key='food_item', info_map=dict()):
        pose = quaternion_matrix(rvec)
        pose[:3, 3] = tvec

        return DetectedItem(
            frame_id=self.frame,
            marker_namespace=t_class_name,
            marker_id=box_id,
            db_key=db_key,
            pose=pose,
            detected_time=rospy.Time.now(),
            bbox=bbox,
            info_map=info_map)

    # Inherited classes change this
    def get_skewering_pose(self, txmin, txmax, tymin, tymax, width, height,
                           img_msg, t_class_name, annotation):
        """
        @return list of skewering position, angle,
        and other information for each detected item in the image.
        """
        # return [[0.5, 0.5]], [0.0], [dict()]
        return [0.5, 0.5], 0.0, dict()
        

    def get_index_of_class_name(self, class_name):
        for index, name in self.label_map.items():
            if name == class_name:
                return index
        return -1

    def calculate_depth(self, depth_img):
        depth = depth_img.flatten()
        depth = depth[depth > 0]
        if depth is None or len(depth) == 0:
            return -1
        depth = depth[abs(depth - np.mean(depth)) < np.std(depth)]

        z0 = np.mean(depth)
        if self.z is None:
            self.z = z0
        else:
            z0 = self.ema_alpha * z0 + (1.0 - self.ema_alpha) * self.z
            self.z = z0
        return z0 / 1000.0  # mm to m

    # Entry point to initialize any annotation function with all boxes
    def annotation_initialization(self, boxes):
        pass

    # Entry point to annotate individual boxes with arbitrary data
    def annotate_box(self, box=None):
        return None

    def detect_objects(self, raw_img=None):
        # print("called")
        if self.img is None and raw_img is None:
            #TODO return [], and a blank img; for simulatiton, it might happen if you do not keep pub img to camera topic
            return list(), self.img

        if self.depth_img is None:
            self.depth_img = np.ones(self.img.shape[:2])

        copied_img = self.img.copy()
        if raw_img is not None:
            # cv2.imshow("raw_img", raw_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print("use img from rqt")
            copied_img = raw_img.copy()
        img = PILImage.fromarray(copied_img.copy())
        depth_img = self.depth_img.copy()

        width, height = img.size

        x = self.retinanet_transform(img)
        x = x.unsqueeze(0)
        with torch.no_grad():
            if self.use_cuda:
                loc_preds, cls_preds = self.retinanet(x.cuda())
            else:
                loc_preds, cls_preds = self.retinanet(x)

            boxes, labels, scores = self.encoder.decode(
                loc_preds.cpu().data.squeeze(),
                cls_preds.cpu().data.squeeze(),
                (width, height))

        if boxes is None or len(boxes) == 0:
            msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
            # self.pub_img.publish(msg_img)
            #TODO deal with food detected case
            return [], msg_img
        
        name_labels = []
        for label in labels:
            t = label.item()
            name_labels.append(self.label_map[t])
        
        bbox_img = self.visualize_detections(img, boxes, scores, name_labels)
        bbox_img_msg = self.bridge.cv2_to_imgmsg(np.array(bbox_img), "rgb8")
        # ---------------------- Pose Estimation in Camera Frame ---------------------- #

        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        # Projects 3D points in the camera coordinate frame to 2D pixel
        # coordinates using the focal lengths (fx, fy) and principal point
        # (cx, cy).
        camera_matrix = np.asarray(self.camera_info.K).reshape(3, 3)
        cam_fx = camera_matrix[0, 0]
        cam_fy = camera_matrix[1, 1]
        cam_cx = camera_matrix[0, 2]
        cam_cy = camera_matrix[1, 2]

        detections = []

        bbox_offset = 5
        for box_idx in range(len(boxes)):
            t = labels[box_idx].item()
            t_class_name_current = self.label_map[t]
            
            # ------------ Translation Vector ------------ #
            txmin, tymin, txmax, tymax = boxes[box_idx].numpy() - bbox_offset
            if (txmin < 0 or tymin < 0 or txmax > width or tymax > height):
                continue
            
            cropped_depth = depth_img[
                int(max(tymin, 0)):int(min(tymax, height)),
                int(max(txmin, 0)):int(min(txmax, width))]
            z0 = self.calculate_depth(cropped_depth)
            if z0 < 0:
                continue

            # ---------------------- SPANet ---------------------- #
            annotation = self.annotate_box()
            skewer_xy, skewer_angle, skewer_info = self.get_skewering_pose(
                    txmin, txmax, tymin, tymax, width, 
                    height, copied_img, t_class_name_current, annotation)

            txoff = (txmax - txmin) * skewer_xy[0]
            tyoff = (tymax - tymin) * skewer_xy[1]
            pt = [txmin + txoff, tymin + tyoff]

            coff = 60
            cropped_depth = depth_img[
                int(pt[1] - coff):int(pt[1] + coff),
                int(pt[0] - coff):int(pt[0] + coff)]
            current_z0 = self.calculate_depth(cropped_depth) # this  cuurrent_z0 is always < 0
            if (current_z0 < 0):
                print("replace it")
                current_z0 = z0
            tz = current_z0
            tx = (tz / cam_fx) * (pt[0] - cam_cx)
            ty = (tz / cam_fy) * (pt[1] - cam_cy)
            tvec = np.array([tx, ty, tz])

            # ------------ Rotation Vector ------------ #
            x, y, z, w = quaternion_from_euler(0., 0., 90. + skewer_angle)
            rvec = np.array([x, y, z, w])

            detections.append(self.create_detected_item(rvec, tvec, t_class_name_current, box_idx, 
                                bbox=boxes[box_idx].numpy().astype(np.int), info_map=skewer_info))
        # print(len(detections), bbox_img.size)
        return detections, bbox_img_msg

    def visualize_detections(self, img, boxes, scores, labels, bbox_offset=5, push_type="no_push"):
        # visualize detections
        fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 12)
        fnt2 = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 20)
        draw = ImageDraw.Draw(img, 'RGBA')

        w, h = img.size
        pw, ph = fnt2.getsize(push_type)
        draw.text(xy=(w-pw, h-ph), text=push_type, fill="red", font=fnt2)

        for idx in range(len(boxes)):
            box = boxes[idx].numpy() - bbox_offset
            label = labels[idx]
            draw.rectangle(box, outline=(255, 0, 0, 200))
            box1 = box + 1
            box1[:2] -= 2
            draw.rectangle(box1, outline=(255, 0, 0, 200))
            box2 = box + 2
            box2[:2] -= 4
            draw.rectangle(box2, outline=(255, 0, 0, 200))

            item_tag = '{0}: {1:.2f}'.format(
                label,
                scores[idx])
            iw, ih = fnt.getsize(item_tag)
            ix, iy = box[:2]
            draw.rectangle((ix, iy, ix + iw, iy + ih), fill=(255, 0, 0, 100))
            draw.text(
                box[:2],
                item_tag,
                font=fnt, fill=(255, 255, 255, 255))

        # msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
        # self.pub_img.publish(msg_img)
        return img
