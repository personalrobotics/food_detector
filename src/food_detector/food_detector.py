#!/usr/bin/python2

from __future__ import print_function
from __future__ import division
from __future__ import with_statement

import numpy as np
import os
import sys
import rospy
import rospkg

from scipy.special import softmax
from sensor_msgs.msg import Image

from PIL import Image as PILImage
from PIL import ImageDraw

import torch
import torchvision.transforms as transforms

import ada_feeding_demo_config as conf

rospack = rospkg.RosPack()
pkg_base = rospack.get_path('food_detector')
external_path = os.path.join(
    pkg_base, 'external')
sys.path.append(external_path)

from bite_selection_package.model.spnet import SPNet, DenseSPNet
from bite_selection_package.config import spnet_config

from retinanet_detector import RetinaNetDetector


# A pose estimator for detecting object and skewering pose
# using SPNet
class FoodDetector(RetinaNetDetector):
    def __init__(self, node_name, use_cuda=True):
        RetinaNetDetector.__init__(
            self,
            node_name=node_name,
            use_cuda=use_cuda,
            retinanet_checkpoint=conf.checkpoint,
            label_map_file=conf.label_map,
            camera_to_table=conf.camera_to_table,
            camera_tilt=1e-5,
            frame=conf.camera_tf)

        self.agg_pc_data = list()

        self.spnet = None
        self.spnet_transform = None

        self.angle_res = spnet_config.angle_res
        self.mask_size = spnet_config.mask_size
        self.use_densenet = spnet_config.use_densenet
        self.target_position = np.array([320, 240])

        self.final_size = 512
        self.target_size = 136

        self.pub_spnet_img = rospy.Publisher(
            '{}/spnet_image'.format(self.node_name),
            Image,
            queue_size=2)

        self.init_spnet()

    def init_spnet(self):
        if self.use_densenet:
            self.spnet = DenseSPNet()
        else:
            self.spnet = SPNet()
        print('Loaded {}SPNet'.format('Dense' if self.use_densenet else ''))

        if self.use_cuda:
            ckpt = torch.load(
                os.path.expanduser(conf.spnet_checkpoint))
        else:
            ckpt = torch.load(
                os.path.expanduser(conf.spnet_checkpoint),
                map_location='cpu')

        self.spnet.load_state_dict(ckpt['net'])
        self.spnet.eval()
        if self.use_cuda:
            self.spnet = self.spnet.cuda()

        self.spnet_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     (0.562, 0.370, 0.271), (0.332, 0.302, 0.281))
        ])

    def publish_spnet(self, sliced_img, identity, actuallyPublish=False):
        should_publish_spnet = True
        if rospy.has_param('/deep_pose/publish_spnet'):
            should_publish_spnet = rospy.get_param('/deep_pose/publish_spnet')
        if not should_publish_spnet:
            return [[0.5, 0.5]], [-90]

        spnet_food_name = 'all'
        if rospy.has_param('/deep_pose/spnet_food_name'):
            spnet_food_name = rospy.get_param('/deep_pose/spnet_food_name')
        if spnet_food_name != 'all' and spnet_food_name != identity:
            return [], []

        invert_spnet_direction = False
        if rospy.has_param('/deep_pose/invertSPNetDirection'):
            invert_spnet_direction = rospy.get_param(
                '/deep_pose/invertSPNetDirection')

        img_org = PILImage.fromarray(sliced_img.copy())

        ratio = float(self.target_size / max(img_org.size))
        new_size = tuple([int(x * ratio) for x in img_org.size])
        pads = [(self.target_size - new_size[0]) // 2,
                (self.target_size - new_size[1]) // 2]
        img_org = img_org.resize(new_size, PILImage.ANTIALIAS)
        img = PILImage.new('RGB', (self.target_size, self.target_size))
        img.paste(img_org, pads)
        transform = transforms.Compose([transforms.ToTensor()])
        pred_bmasks, pred_rmasks = self.spnet(
            torch.stack([transform(img)]).cuda())

        np.set_printoptions(
            edgeitems=30, linewidth=100000,
            formatter=dict(float=lambda x: "%.3g" % x))

        img = img.resize((self.final_size, self.final_size),
                         PILImage.ANTIALIAS)
        draw = ImageDraw.Draw(img, 'RGBA')

        bmask = pred_bmasks[0].data.cpu().numpy()
        bmask = softmax(bmask)
        neg_pos = bmask < 0.001

        rmask = pred_rmasks[0].data.cpu().numpy()
        rmask = softmax(rmask, axis=1)
        neg_rot = np.max(rmask, axis=1) < 0.5

        rmask_prob = np.max(rmask, axis=1)
        rmask_prob = rmask_prob.reshape(self.mask_size, self.mask_size)

        rmask_argmax = np.argmax(rmask, axis=1) - 1

        rmask = rmask_argmax * 180 / self.angle_res

        if identity in ["celery", "carrot"]:
            rmask[rmask < 0] = -2  # don't use "no rotation"
        else:
            rmask[rmask < 0] = -1  # do use "no rotation"
        rmask[neg_pos] = -2
        rmask[neg_rot] = -2
        rmask = rmask.reshape(self.mask_size, self.mask_size)

        bmask = bmask.reshape(self.mask_size, self.mask_size)

        rotation_mode = 'normal'  # normal / alt
        sp_poses = list()
        sp_angles = list()
        sp_scores = list()

        done = False
        for ri in range(self.mask_size):
            for ci in range(self.mask_size):
                rotation = rmask[ri][ci]
                score = bmask[ri][ci]
                if rotation >= -1:
                    rotation = -rotation
                    ix = ci * self.final_size / self.mask_size
                    iy = ri * self.final_size / self.mask_size

                    rot_rad = np.radians(rotation)
                    iw = (-np.sin(rot_rad) * 4 *
                          (self.final_size / self.target_size))
                    ih = (np.cos(rot_rad) * 4 *
                          (self.final_size / self.target_size))

                    rot_alpha = int(bmask[ri][ci] * 200) + 55
                    if -rotation == -1:
                        line_color = (40, 255, 100, rot_alpha)
                    else:
                        line_color = (30, 30, 250, rot_alpha)

                    draw.line(
                        (ix - iw, iy - ih, ix + iw, iy + ih),
                        fill=line_color,
                        width=int(float(self.final_size) /
                                  float(self.target_size)*2))
                    sp_poses.append(
                        [ci / float(self.mask_size),
                         ri / float(self.mask_size)])
                    sp_scores.append(score)

                    x1 = iw
                    y1 = ih
                    x2 = 0.5 - ci / float(self.mask_size)
                    y2 = 0.5 - ri / float(self.mask_size)
                    a = x1 * y2 - x2 * y1
                    if invert_spnet_direction:
                        if a < 0:
                            rotation += 180
                    else:
                        if a > 0:
                            rotation += 180
                    sp_angles.append(rotation)
                    # done = True
                if done:
                    break
            if done:
                break

        if actuallyPublish:
            msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
            self.pub_spnet_img.publish(msg_img)

        if not sp_scores:
            return None, None

        # Return only the one with the highest score
        max_score_idx = np.argmax(sp_scores)
        return sp_poses[max_score_idx], sp_angles[max_score_idx]

    def get_skewering_pose(
            self, txmin, txmax, tymin, tymax, width,
            height, img_msg, t_class_name):
        """
        @return skewering position and angle in the image.
        """
        """
        Temporarily Disable for Demo
        cropped_img = img_msg[int(max(tymin, 0)):int(min(tymax, height)),
                              int(max(txmin, 0)):int(min(txmax, width))]

        sp_pose, sp_angle = self.publish_spnet(cropped_img, t_class_name, True)
        return sp_pose, sp_angle
        """

        return [0.5, 0.5], 0.0
