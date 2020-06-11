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
from bite_selection_package.utils.visualize_spanet import draw_image

from retinanet_detector import RetinaNetDetector
from image_publisher import ImagePublisher
import spa_demo_config as conf
from wall_detector import WallDetector, WallClass
N_FEATURES = 2048 if spanet_config.n_features==None else spanet_config.n_features
ACTIONS = ['vertical', 'tilted-vertical', 'tilted-angled']

class SPANetDetector(RetinaNetDetector):
    """
    Action detector returns particular action as the class of each object.
    """

    def __init__(self, use_cuda=True, use_walldetector=True,
        num_action_per_item=1):
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

        print("Use wall detector: ", use_walldetector)
        self.use_walldetector = use_walldetector

        if self.use_walldetector:
            self.wall_detector = WallDetector() # Toggle debug
        else:
            self.wall_detector = None

        self.use_densenet = spanet_config.use_densenet
        self.target_position = np.array([320, 240])

        self.final_size = N_FEATURES  # DEBUG: XM: Changed this size from 512 to N_FEATURES
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
            self.spanet = SPANet(use_wall=self.use_walldetector)
        print('Loaded {}SPANet'.format('Dense' if self.use_densenet else ''))

        if self.use_cuda:
            print("Use cuda")
            if not self.use_walldetector:
                print("-----------------------------------" + conf.spanet_checkpoint)
                ckpt = torch.load(
                    os.path.expanduser(conf.spanet_checkpoint))
            else:
                ckpt = torch.load(
                    os.path.expanduser(conf.spanet_wall_checkpoint))
        else:
            if not self.use_walldetector:
                ckpt = torch.load(
                    os.path.expanduser(conf.spanet_checkpoint),
                    map_location='cpu')
            else:
                ckpt = torch.load(
                    os.path.expanduser(conf.spanet_wall_checkpoint),
                    map_location='cpu')

        self.spanet.load_state_dict(ckpt['net'])
        self.spanet.eval()
        if self.use_cuda:
            self.spanet = self.spanet.cuda()

        self.spanet_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # Entry point to initialize any annotation function with all boxes
    def annotation_initialization(self, boxes):
        if self.wall_detector is None:
            return None
        # Register all UV Points in wall detector
        self.wall_detector.register_items(boxes)


    # Entry point to annotate individual boxes with arbitrary data
    def annotate_box(self, box):
        if self.wall_detector is None:
            return None

        wall_type = WallClass.kISOLATED #self.wall_detector.classify(box, self.img_msg, self.depth_img_msg)

        if wall_type == WallClass.kNEAR_OBJ:
            return dict(tensor=torch.tensor([[0., 1., 0.]]), type=WallClass.kNEAR_OBJ)
        elif wall_type == WallClass.kON_OBJ:
            return dict(tensor=torch.tensor([[0., 0., 1.]]), type=WallClass.kON_OBJ)

        return dict(tensor=torch.tensor([[1., 0., 0.]]), type=WallClass.kISOLATED)

    def get_skewering_pose(
            self, txmin, txmax, tymin, tymax, width,
            height, img_msg, t_class_name, annotation):
        """
        @return list of skewering position, angle,
        and other information for each detected item in the image.
        """
        cropped_img = img_msg[int(max(tymin, 0)):int(min(tymax, height)),
                              int(max(txmin, 0)):int(min(txmax, width))]
        for dim in cropped_img.shape:
            if dim == 0:
                return None, None, None
        positions, angles, actions, scores, rotations, features = self.publish_spanet(
            cropped_img, t_class_name, True, annotation['tensor'])

        info_maps = [dict(
            features=features,
            action=action,
            uv=[float(txmin + txmax) / 2.0, float(tymin + tymax) / 2.0],
            score=round(float(score),2),
            annotation=str(annotation['type']),
            rotation=float(rotation)) for rotation, angle, action, score in zip(
                rotations, angles, actions, scores)]

        return positions, angles, info_maps

    def publish_spanet(self, sliced_img, identity, actuallyPublish=False, loc_type=None):
        img_org = PILImage.fromarray(sliced_img.copy())
        ratio = float(self.target_size / max(img_org.size))
        new_size = tuple([int(x * ratio) for x in img_org.size])
        pads = [(self.target_size - new_size[0]) // 2,
                (self.target_size - new_size[1]) // 2]
        img_org = img_org.resize(new_size, PILImage.ANTIALIAS)
        img = PILImage.new('RGB', (self.target_size, self.target_size))
        img.paste(img_org, pads)
        transform = transforms.Compose([transforms.ToTensor()])
        pred_vector, features = self.spanet(
            torch.stack([transform(img).cuda()]), None, loc_type.cuda())

        pred_vector = pred_vector.cpu().detach().numpy().flatten()
        features_flat = features.cpu().detach().numpy().flatten().tolist()
        # Add Bias
        features_flat.insert(0, 1.0)

        # pred_vector: [p1_row, p1_col, p2_row, p2_col, a1_success_rate, ..., a6_suceess_rate]
        p1 = pred_vector[:2]
        p2 = pred_vector[2:4]

        position = np.divide(p1 + p2, 2.0)

        # Offset if necessary
        # fudge_offset = np.array([[-0.2, 0.8]]).reshape(position.shape)
        # position = position + fudge_offset
        # print("Position After: ", position)

        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

        success_rates = pred_vector[4:]
        order = np.argsort(success_rates * -1)

        self.visualize_spanet(img, pred_vector)

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

        if not rospy.has_param('/spanetIncludeFeatures'):
            features_flat = None
        elif not rospy.get_param('/spanetIncludeFeatures'):
            features_flat = None

        return [positions[0]], [angles[0]], [action_names[0]], [best_success_rates[0]], [rotations[0]], features_flat

    def visualize_spanet(self, image, pred_vector):
        img = draw_image(image, pred_vector)
        msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
        self.pub_spanet_img.publish(msg_img)
