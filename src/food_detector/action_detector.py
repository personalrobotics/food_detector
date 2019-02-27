#!/usr/bin/env python

import os
import json
import numpy as np
import rospy
from tf import TransformListener
from sensor_msgs.msg import Image
import torch
import torchvision.transforms as transforms

from pose_estimators.pose_estimators import PoseEstimator
from pose_estimators.detected_item import DetectedItem
from pose_estimators.utils.ros_utils import get_transform_matrix
from pose_estimators.utils import CameraSubscriber

from bite_selection_package.model.spanet import SPANet
from bite_selection_package.config import spanet_config

from retinanet_detector import RetinaNetDetector
from image_publisher import ImagePublisher
import spa_demo_config as conf

ACTIONS = ['vertical', 'tilted-vertical', 'tilted-angled']

class ActionDetector(RetinaNetDetector):
    """
    Action detector returns particular action as the class of each object.
    """

    def __init__(self, use_cuda=True, use_walldetector=False):
        RetinaNetDetector.__init__(
            self,
            retinanet_checkpoint=conf.checkpoint,
            use_cuda=use_cuda,
            label_map_file=conf.label_map,
            node_name=conf.node_name,
            camera_to_table=conf.camera_to_table,
            camera_tilt=1e-5,
            frame=conf.camera_tf)

        #self.score = json.loads(scores)
        #self.actions = ["vertical_skewer_0",
        #                "vertical_tilted_skewer_0",
        #                "vertical_angled_skewer_0"]

        self.listener = TransformListener()
        self.detection_frame = conf.camera_tf
        self.destination_frame = conf.destination_frame
        self.timeout = 1.0

        self.use_walldetector = False

        if self.use_walldetector:
            self.wall_detector = WallDetector()
            # TODO load scores map and actions
            self.scores = dict()
            self.actions = []
        else:
            self.wall_detector = None

        self.agg_pc_data = list()

        self.use_densenet = spanet_config.use_densenet
        self.target_position = np.array([320, 240])

        self.final_size = 512
        self.target_size = 136

        self.pub_img = rospy.Publisher(
            '{}/detection_image'.format(self.node_name),
            Image,
            queue_size=2)
        self.pub_target_img = rospy.Publisher(
            '{}/target_image'.format(self.node_name),
            Image,
            queue_size=2)
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

        # Get the transform from destination to detection frame
        camera_transform = get_transform_matrix(self.listener,
            self.destination_frame,
            self.detection_frame,
            self.timeout)

        # Register all UV Points in wall detector
        for item in detected_items:
            self.wall_detector.register_uv(item.info_map['uv'])

        for item in detected_items:
            wall_type = self.wall_detector.classify(uv, self.img_msg, self.depth_img_msg)

            scores = [self.score[item.namespace][wall_type][action]
                for action in self.actions]

            best_action = self.actions[np.argmax(scores)]

            item.info_map["best_action"] = best_action
            item.info_map["best_action_score"] = score

        return item

    def get_skewering_pose(
            self, txmin, txmax, tymin, tymax, width,
            height, img_msg, t_class_name):
        """
        @return skewering position and angle in the image.
        """
        cropped_img = img_msg[int(max(tymin, 0)):int(min(tymax, height)),
                              int(max(txmin, 0)):int(min(txmax, width))]
        position, angle, action = self.publish_spanet(cropped_img, t_class_name, True)

        return position, angle, dict(action=action,
            uv=((txmin + txmax) / 2.0, (tymin + tymax) / 2.0))

    def publish_spanet(self, sliced_img, identity, actuallyPublish=False):

        pred_vector, _ = self.spanet(img)

        pred_vector = pred_vector.cpu().detach().numpy()

        # pred_vector: [p1_row, p1_col, p2_row, p2_col, a1_success_rate, ..., a6_suceess_rate]
        p1 = pred_vector[:2]
        p2 = pred_vector[2:4]

        point = np.mean(p1, p2)
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        success_rates = pred_vector[4:]
        action_idx = np.argmax(success_rates)

        if (action_idx % 2 == 0):
            angle = 0
        else:
            angle = 90

        action_name = ACTIONS[action_idx // 2]

        return position, angle, action_name

