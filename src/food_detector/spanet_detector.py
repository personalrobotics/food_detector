#!/usr/bin/env python
from __future__ import division, print_function

import os
import json
import numpy as np
import rospy
from tf import TransformListener
from sensor_msgs.msg import Image
import torch
import torchvision.transforms as transforms

from .wall_detector import WallDetector, WallClass

from PIL import Image as PILImage

from pose_estimators.pose_estimator import PoseEstimator
from pose_estimators.detected_item import DetectedItem
from pose_estimators.utils.ros_utils import get_transform_matrix
from pose_estimators.utils import CameraSubscriber

from bite_selection_package.model.spanet import SPANet
from bite_selection_package.model.spanet import DenseSPANet
from bite_selection_package.config import spanet_config
#from bite_selection_package.utils.visualize_spanet import draw_image

from retinanet_detector import RetinaNetDetector
from image_publisher import ImagePublisher
import spa_demo_config as conf

ACTIONS = ['vertical', 'tilted-vertical', 'tilted-angled']

class SPANetDetector(RetinaNetDetector):
    """
    Action detector returns particular action as the class of each object.
    """

    def __init__(self, use_cuda=True, use_walldetector=True,
        num_action_per_item=2):
        RetinaNetDetector.__init__(
            self,
            retinanet_checkpoint=conf.checkpoint,
            use_cuda=use_cuda,
            label_map_file=conf.label_map,
            node_name=conf.node_name,
            camera_to_table=conf.camera_to_table,
            camera_tilt=1e-5,
            frame=conf.camera_tf)

        self.listener = TransformListener()
        self.detection_frame = conf.camera_tf
        self.destination_frame = conf.destination_frame
        self.timeout = 1.0

        self.use_walldetector = use_walldetector

        if self.use_walldetector:
            self.wall_detector = WallDetector()
        else:
            self.wall_detector = None

        self.use_densenet = spanet_config.use_densenet
        self.target_position = np.array([320, 240])

        self.final_size = 512
        self.target_size = 144

        self.num_action_per_item = num_action_per_item

        self.pub_spanet_img = rospy.Publisher(
            '{}/spanet_image'.format(self.node_name),
            Image,
            queue_size=2)

        self.init_spanet()

    def init_spanet(self):
        if self.use_densenet:
            self.spanet = DenseSPANet()
        else:
            self.spanet = SPANet()
        print('Loaded {}SPANet'.format('Dense' if self.use_densenet else ''))

        if self.use_cuda:
            ckpt = torch.load(
                os.path.expanduser(conf.spanet_checkpoint))
        else:
            ckpt = torch.load(
                os.path.expanduser(conf.spanet_checkpoint),
                map_location='cpu')

        self.spanet.load_state_dict(ckpt['net'])
        self.spanet.eval()
        if self.use_cuda:
            self.spanet = self.spanet.cuda()

        self.spanet_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def detect_objects(self):
        # Get DetectedItems using SPANet
        detected_items = RetinaNetDetector.detect_objects(self)

        if self.wall_detector is None:
            return detected_items

        # Register all UV Points in wall detector
        self.wall_detector.register_items(detected_items)

        for item in detected_items:
            wall_type = self.wall_detector.classify(item, self.img_msg, self.depth_img_msg)

            if wall_type != WallClass.kUNKNOWN:
                print("Item ID: %d" % item.marker_id)
                print("Wall Type: " + str(wall_type))
                print()

        return detected_items

    def get_skewering_pose(
            self, txmin, txmax, tymin, tymax, width,
            height, img_msg, t_class_name):
        """
        @return list of skewering position, angle,
        and other information for each detected item in the image.
        """
        cropped_img = img_msg[int(max(tymin, 0)):int(min(tymax, height)),
                              int(max(txmin, 0)):int(min(txmax, width))]
        for dim in cropped_img.shape:
            if dim == 0:
                return None, None, None
        positions, angles, actions, scores, rotations = self.publish_spanet(
            cropped_img, t_class_name, True)

        info_maps = [dict(
            action=action,
            uv=[float(txmin + txmax) / 2.0, float(tymin + tymax) / 2.0],
            score=round(float(score),2),
            rotation=float(rotation)) for rotation, action, score in zip(
                rotations, actions, scores)]

        return positions, angles, info_maps

    def publish_spanet(self, sliced_img, identity, actuallyPublish=False):
        img_org = PILImage.fromarray(sliced_img.copy())
        ratio = float(self.target_size / max(img_org.size))
        new_size = tuple([int(x * ratio) for x in img_org.size])
        pads = [(self.target_size - new_size[0]) // 2,
                (self.target_size - new_size[1]) // 2]
        img_org = img_org.resize(new_size, PILImage.ANTIALIAS)
        img = PILImage.new('RGB', (self.target_size, self.target_size))
        img.paste(img_org, pads)
        transform = transforms.Compose([transforms.ToTensor()])
        pred_vector, _ = self.spanet(
            torch.stack([transform(img).cuda()]), None)

        pred_vector = pred_vector.cpu().detach().numpy().flatten()

        # pred_vector: [p1_row, p1_col, p2_row, p2_col, a1_success_rate, ..., a6_suceess_rate]
        p1 = pred_vector[:2]
        p2 = pred_vector[2:4]

        position = np.divide(p1 + p2, 2.0)
        # Offset if necessary
        fudge_offset = np.array([[0, 0]]).reshape(position.shape)
        position = position + fudge_offset
        print("Position: " + str(position))

        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

        success_rates = pred_vector[4:]
        order = np.argsort(success_rates * -1)

#        self.visualize_spanet(img, pred_vector)

        positions = [position] * self.num_action_per_item
        angles = [angle] * self.num_action_per_item

        rotations = []
        action_names = []
        best_success_rates = []

        for action_idx in order[:self.num_action_per_item]:
            success_rate = success_rates[action_idx]

            if (action_idx % 2 == 0):
                rotation = 0
            else:
                rotation = 90.0

            action_names += [ACTIONS[action_idx // 2]]
            best_success_rates += [success_rate]
            rotations += [rotation]

        return positions, angles, action_names, best_success_rates, rotations

#    def visualize_spanet(self, image, pred_vector):
#        img = draw_image(image, pred_vector)
#        msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
#        self.pub_spanet_img.publish(msg_img)
