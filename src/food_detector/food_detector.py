#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import with_statement

import numpy as np
import os
import sys
import json
import cv2
import pcl
import rospy
import rospkg

from tf.transformations import quaternion_matrix, quaternion_from_euler
from scipy.special import softmax
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge

import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

import ada_feeding_demo_config as conf

rospack = rospkg.RosPack()
pkg_base = rospack.get_path('food_detector')
external_path = os.path.join(
    pkg_base, 'external')
sys.path.append(external_path)

from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.retinanet_utils.encoder import DataEncoder

from bite_selection_package.model.spnet import SPNet, DenseSPNet
from bite_selection_package.spnet_config import config as spnet_config

from laura_model1.run_test import Model1

from deep_pose_estimators.pose_estimators import PoseEstimator
from deep_pose_estimators.utils import pcl_utils
from deep_pose_estimators.detected_item import DetectedItem


# A pose estimator for detecting object and skewering pose
class FoodDetector(PoseEstimator):
    def __init__(self, title='food_detector',
                 use_spnet=True, use_cuda=True, use_model1=False):
        self.title = title

        self.img_msg = None
        self.depth_img_msg = None
        self.retinanet = None
        self.retinanet_transform = None
        self.label_map = None
        self.encoder = None

        self.use_cuda = use_cuda

        self.agg_pc_data = list()
        self.camera_to_table = conf.camera_to_table
        self.frame = conf.camera_tf

        self.use_spnet = use_spnet
        self.spnet = None
        self.spnet_transform = None

        self.use_model1 = use_model1
        self.model1 = None
        self.model1_transform = None

        self.angle_res = spnet_config.angle_res
        self.mask_size = spnet_config.mask_size
        self.use_densenet = spnet_config.use_densenet
        self.target_position = np.array([320, 240])

        self.final_size = 512
        self.target_size = 136

        self.camera_tilt = 1e-5

        self.init_ros_subscribers()
        self.init_ros_publishers()

        self.pub_img = rospy.Publisher(
            '{}/detection_image'.format(self.title),
            Image,
            queue_size=2)
        self.pub_target_img = rospy.Publisher(
            '{}/target_image'.format(self.title),
            Image,
            queue_size=2)
        self.pub_spnet_img = rospy.Publisher(
            '{}/spnet_image'.format(self.title),
            Image,
            queue_size=2)

        self.bridge = CvBridge()

        self.selector_food_names = [
            "carrot", "melon", "apple", "banana", "strawberry"]
        self.selector_index = 0

    def create_detected_item(self, rvec, tvec, t_class_name,
                             db_key='food_item'):
        pose = quaternion_matrix(rvec)
        pose[:3, 3] = tvec

        return DetectedItem(
            frame_id=self.frame,
            marker_namespace='{}_{}'.format(self.title, t_class_name),
            marker_id=-1,  # It is the marker manager's job to assign an id
            db_key=db_key,
            pose=pose,
            detected_time=rospy.Time.now())

    def init_ros_subscribers(self):
        # subscribe image topic
        if conf.msg_type == 'compressed':
            self.img_subscriber = rospy.Subscriber(
                conf.image_topic, CompressedImage,
                self.sensor_compressed_image_callback, queue_size=1)
        else:  # raw
            self.img_subscriber = rospy.Subscriber(
                conf.image_topic, Image,
                self.sensor_image_callback, queue_size=1)
        print('subscribed to {}'.format(conf.image_topic))

        # subscribe depth topic
        if (conf.depth_image_topic is not None and
                len(conf.depth_image_topic) > 0):
            # subscribe depth topic, only raw for now
            self.depth_subscriber = rospy.Subscriber(
                conf.depth_image_topic, Image,
                self.sensor_depth_callback, queue_size=1)
            print('subscribed to {}'.format(conf.depth_image_topic))

        # subscribe camera info topic
        self.subscriber = rospy.Subscriber(
                conf.camera_info_topic, CameraInfo,
                self.camera_info_callback)
        print('subscribed to {}'.format(conf.camera_info_topic))

    def init_ros_publishers(self):
        self.pub_img = rospy.Publisher(
            '{}/detection_image'.format(self.title),
            Image,
            queue_size=2)

    def sensor_compressed_image_callback(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        new_msg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img_msg = cv2.cvtColor(new_msg, cv2.COLOR_BGR2RGB)

    def sensor_image_callback(self, ros_data):
        self.img_msg = self.bridge.imgmsg_to_cv2(ros_data, 'rgb8')

    def sensor_depth_callback(self, ros_data):
        self.depth_img_msg = self.bridge.imgmsg_to_cv2(ros_data, '16UC1')

    def camera_info_callback(self, ros_data):
        self.camera_info = ros_data

    def init_retinanet(self):
        self.retinanet = RetinaNet()
        if self.use_cuda:
            ckpt = torch.load(os.path.expanduser(conf.checkpoint))
        else:
            ckpt = torch.load(
                os.path.expanduser(conf.checkpoint), map_location='cpu')
        self.retinanet.load_state_dict(ckpt['net'])
        self.retinanet.eval()
        if self.use_cuda:
            self.retinanet = self.retinanet.cuda()

        print('Loaded RetinaNet.')

        self.retinanet_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.encoder = DataEncoder()

    def init_spnet(self):
        if self.use_densenet:
            self.spnet = DenseSPNet()
        else:
            self.spnet = SPNet()
        print('Loaded {}SPNet'.format('Dense' if self.use_densenet else ''))

        if self.use_cuda:
            ckpt = torch.load(
                os.path.expanduser(spnet_config.checkpoint_best_filename))
        else:
            ckpt = torch.load(
                os.path.expanduser(spnet_config.checkpoint_best_filename),
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

    def init_model1(self):
        self.model1 = Model1()

        if self.use_cuda:
            ckpt = torch.load(os.path.join(
                    os.path.dirname(__file__),
                    'external/laura_model1/checkpoint/model1_ckpt.pth'))
        else:
            ckpt = torch.load(os.path.join(
                    os.path.dirname(__file__),
                    'external/laura_model1/checkpoint/model1_ckpt.pth'),
                map_location='cpu')

        self.model1.load_state_dict(ckpt['net'])
        self.model1.eval()
        if self.use_cuda:
            self.model1 = self.model1.cuda()

        print('Loaded Laura\'s Model1')

        self.model1_transform = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
        ])

    def load_label_map(self):
        with open(os.path.expanduser(conf.label_map), 'r') as f:
            content = f.read().splitlines()
            f.close()
        assert content is not None, 'cannot find label map'

        temp = list()
        for line in content:
            line = line.strip()
            if (len(line) > 2 and
                    (line.startswith('id') or
                     line.startswith('name'))):
                temp.append(line.split(':')[1].strip())

        label_dict = dict()
        for idx in range(0, len(temp), 2):
            item_id = int(temp[idx])
            item_name = temp[idx + 1][1:-1]
            label_dict[item_id] = item_name

        self.label_map = label_dict

    def get_index_of_class_name(self, class_name):
        for index, name in self.label_map.items():
            if name == class_name:
                return index
        return -1

    def get_box_coordinates(self, box, img_shape):
        txmin = int(box[0] * img_shape[0])
        tymin = int(box[1] * img_shape[1])
        txmax = int(box[2] * img_shape[0])
        tymax = int(box[3] * img_shape[1])
        return txmin, tymin, txmax, tymax

    def calculate_depth_from_depth_image(self, xmin, ymin, xmax, ymax, dimg):
        dimg_sliced = np.array(dimg)[int(xmin):int(xmax), int(ymin):int(ymax)]
        depth = dimg_sliced.flatten()
        depth = depth[depth > 0]
        if depth is None or len(depth) == 0:
            return -1
        z0 = np.mean(depth)
        return z0 / 1000.0  # mm to m

    def calculate_depth(self, depth_img):
        depth = depth_img.flatten()
        depth = depth[depth > 0]
        depth = depth[abs(depth - np.mean(depth)) < np.std(depth)]
        if depth is None or len(depth) == 0:
            return -1
        z0 = np.mean(depth)
        return z0 / 1000.0  # mm to m

    def add_group_item(self, grid, x, y, gidx):
        if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
            return False
        if grid[x, y] != 1:
            return False

        grid[x, y] = gidx
        return True

    def propagate_group(self, grid, x, y, gidx):
        next_steps = [
            [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1],
            [x + 1, y + 1], [x - 1, y - 1], [x - 1, y + 1], [x + 1, y - 1]]
        for item in next_steps:
            if self.add_group_item(grid, item[0], item[1], gidx):
                self.propagate_group(grid, item[0], item[1], gidx)

    def group_rmask(self, rmask, rmask_prob):
        grid = np.ones_like(rmask)
        grid[rmask == -2] = 0

        gidx = 2
        for ri in range(grid.shape[0]):
            for ci in range(grid.shape[1]):
                if self.add_group_item(grid, ri, ci, gidx):
                    self.propagate_group(grid, ri, ci, gidx)
                    gidx += 1

        group_list = list()
        for gi in range(2, gidx):
            this_item = list()

            this_group = np.where(grid == gi)
            locations = np.dstack(this_group)[0]
            loc_mean = np.mean(locations, axis=0)
            this_item.append(loc_mean)

            rotations = rmask[this_group]
            rot_prob = rmask_prob[this_group]

            positives = rotations > -1
            rotations = rotations[positives]
            rot_prob = rot_prob[positives]
            if len(rotations) > 0:
                if len(rot_prob) == 1:
                    final_rotation = rotations[0]
                else:
                    final_rotation = np.sum(
                        rotations * softmax(rot_prob))
            else:
                final_rotation = -1
            this_item.append(final_rotation)

            group_list.append(this_item)
        return group_list

    def publish_spnet(self, sliced_img, identity, actuallyPublish = False):
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
            rmask[rmask < 0] = -2 # don't use "no rotation"
        else:
            rmask[rmask < 0] = -1 # do use "no rotation"
        rmask[neg_pos] = -2
        rmask[neg_rot] = -2
        rmask = rmask.reshape(self.mask_size, self.mask_size)

        bmask = bmask.reshape(self.mask_size, self.mask_size)

        sp_mode = 'mask'  # group / mask
        rotation_mode = 'normal'  # normal / alt
        sp_poses = list()
        sp_angles = list()

        if sp_mode == 'group':
            group_list = self.group_rmask(rmask, rmask_prob)

            for item in group_list:
                ri, ci = item[0]
                rotation = item[1]

                if rotation_mode == 'alt':
                    cp = self.mask_size / 2
                    rotation = -np.degrees(np.arctan2(ri - cp, ci - cp))

                rotation = -rotation
                ix = ci * self.final_size / self.mask_size
                iy = ri * self.final_size / self.mask_size

                rot_rad = np.radians(rotation)
                iw = (-np.sin(rot_rad) * 4 *
                      (self.final_size / self.target_size))
                ih = (np.cos(rot_rad) * 4 *
                      (self.final_size / self.target_size))

                if -rotation == -1:
                    line_color = (40, 255, 100, 250)
                else:
                    line_color = (30, 30, 250, 250)

                draw.line(
                    (ix - iw, iy - ih, ix + iw, iy + ih),
                    fill=line_color,
                    width=int(float(self.final_size) /
                              float(self.target_size) * 2))
                sp_poses.append(
                    [ci / float(self.mask_size),
                     ri / float(self.mask_size)])

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

        else:  # sp_mode = 'mask'
            done = False
            for ri in range(self.mask_size):
                for ci in range(self.mask_size):
                    rotation = rmask[ri][ci]
                    if rotation >= -1:
                        if rotation_mode == 'alt':
                            cp = self.mask_size / 2
                            rotation = -np.degrees(
                                np.arctan2(ri - cp, ci - cp))

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
                    if done: break
                if done: break

        if actuallyPublish:
            msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
            self.pub_spnet_img.publish(msg_img)
        return sp_poses, sp_angles

    def detect_objects(self):
        if self.img_msg is None:
            print('no input stream')
            return list()

        if self.depth_img_msg is None:
            print('no input depth stream')
            self.depth_img_msg = np.ones(self.img_msg.shape[:2])

        if self.use_model1:
            if self.model1 is None:
                self.init_model1()
        else:
            if self.retinanet is None:
                self.init_retinanet()

        if self.use_spnet and self.spnet is None:
            self.init_spnet()

        if self.label_map is None:
            self.load_label_map()

        copied_img_msg = self.img_msg.copy()
        img = PILImage.fromarray(copied_img_msg.copy())
        depth_img = self.depth_img_msg.copy()

        width, height = img.size

        force_food = False
        force_food_name = "testberry"
        if (rospy.has_param('/deep_pose/forceFood') and
                rospy.has_param('/deep_pose/forceFoodName')):
            force_food = rospy.get_param('/deep_pose/forceFood')
            force_food_name = rospy.get_param('/deep_pose/forceFoodName')

        if self.use_model1:
            rmax, cmax = img.size
            crop_size = 70
            step_size = 20

            boxes = list()
            labels = list()
            scores = list()

            for ri in range(0, rmax - crop_size, step_size):
                for ci in range(0, cmax - crop_size, step_size):
                    this_cropped_img = img.crop(
                        (ri, ci, ri + crop_size, ci + crop_size))

                    x = self.model1_transform(this_cropped_img)
                    x = x.unsqueeze(0)
                    with torch.no_grad():
                        if self.use_cuda:
                            x = x.cuda()
                        score = torch.sigmoid(
                            self.model1(x)).cpu().data.squeeze()

                        if score > 0.55:
                            boxes.append(
                                [ri, ci, ri + crop_size, ci + crop_size])
                            labels.append(-1)
                            scores.append(score)
            boxes = torch.Tensor(boxes)
            labels = torch.Tensor(labels)
            scores = torch.Tensor(scores)

        else:
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

        sp_poses = [[0.5, 0.5]]
        sp_angles = [-90]

        if boxes is None or len(boxes) == 0:
            msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
            self.pub_img.publish(msg_img)
            return list()

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

        z0 = (self.camera_to_table /
              (np.cos(np.radians(90 - conf.camera_tilt)) + 1e-10))
        # z0 = self.camera_to_table

        detections = list()

        bbox_offset = 5

        first_food_item = True

        found = False
        spBoxIdx = -1
        for _ in range(len(self.selector_food_names)):
            for box_idx in range(len(boxes)):
                t_class = labels[box_idx].item()
                if t_class == -1:
                    t_class_name = 'sample'
                else:
                    t_class_name = self.label_map[t_class]

                if force_food:
                    t_class_name = force_food_name
                    t_class = self.get_index_of_class_name(t_class_name)

                if (t_class_name == self.selector_food_names[self.selector_index]):
                    txmin, tymin, txmax, tymax = boxes[box_idx].numpy() - bbox_offset
                    if (txmin < 0 or tymin < 0 or txmax > width or tymax > height):
                        continue
                    found = True
                    spBoxIdx = box_idx
                    cropped_img = copied_img_msg[
                        int(max(tymin,0)):int(min(tymax, height)),
                        int(max(txmin,0)):int(min(txmax, width))]

                    if self.use_spnet:
                        sp_poses, sp_angles = self.publish_spnet(
                            cropped_img, t_class_name, True)
                    self.last_class_name = t_class_name
                    break

            self.selector_index = ((self.selector_index + 1) %
                                   len(self.selector_food_names))
            if found:
                break

        for box_idx in range(len(boxes)):
            t_class = labels[box_idx].item()
            if t_class == -1:
                t_class_name = 'sample'
            else:
                t_class_name = self.label_map[t_class]
            if force_food:
                t_class_name = force_food_name
                t_class = self.get_index_of_class_name(t_class_name)

            txmin, tymin, txmax, tymax = boxes[box_idx].numpy() - bbox_offset
            if (txmin < 0 or tymin < 0 or txmax > width or tymax > height):
                continue

            cropped_img = copied_img_msg[
                int(max(tymin,0)):int(min(tymax, height)),
                int(max(txmin,0)):int(min(txmax, width))]

            if self.use_spnet:
                sp_poses, sp_angles = self.publish_spnet(
                    cropped_img, t_class_name, False)

            cropped_depth = depth_img[
                int(max(tymin,0)):int(min(tymax, height)),
                int(max(txmin,0)):int(min(txmax, width))]
            z0 = self.calculate_depth(cropped_depth)
            if z0 < 0:
                print("skipping " + t_class_name + " due to invalid z0")
                continue

            if spBoxIdx >= 0:
                for sp_idx in range(len(sp_poses)):
                    # box_key = '{}_{}_{}_{}'.format(
                    #     t_class_name, int(txmin), int(tymin), sp_idx)
                    this_pos = sp_poses[sp_idx]
                    this_ang = sp_angles[sp_idx]

                    txoff = (txmax - txmin) * this_pos[0]
                    tyoff = (tymax - tymin) * this_pos[1]
                    pt = [txmin + txoff, tymin + tyoff]

                    coff = 60
                    cropped_depth = depth_img[
                        int(pt[1]-coff):int(pt[1]+coff),
                        int(pt[0]-coff):int(pt[1]+coff)]
                    current_z0 = self.calculate_depth(cropped_depth)
                    if (current_z0 < 0):
                        current_z0 = z0

                    x, y, z, w = quaternion_from_euler(
                        this_ang + 90, 0., 0.)
                    rvec = np.array([x, y, z, w])

                    tz = current_z0
                    tx = (tz / cam_fx) * (pt[0] - cam_cx)
                    ty = (tz / cam_fy) * (pt[1] - cam_cy)
                    tvec = np.array([tx, ty, tz])

                    detections.append(self.create_detected_item(
                        rvec, tvec, t_class_name, t_class))

        # visualize detections
        fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 12)
        draw = ImageDraw.Draw(img, 'RGBA')

        for idx in range(len(boxes)):
            box = boxes[idx].numpy() - bbox_offset
            label = labels[idx]
            food_name = self.label_map[label.item()]
            if force_food:
                food_name = force_food_name

            draw.rectangle(box, outline=(255, 0, 0, 200))
            box1 = box + 1
            box1[:2] -= 2
            draw.rectangle(box1, outline=(255, 0, 0, 200))
            box2 = box + 2
            box2[:2] -= 4
            draw.rectangle(box2, outline=(255, 0, 0, 200))

            item_tag = '{0}: {1:.2f}'.format(
                food_name,
                scores[idx])
            iw, ih = fnt.getsize(item_tag)
            ix, iy = box[:2]
            draw.rectangle((ix, iy, ix + iw, iy + ih), fill=(255, 0, 0, 100))
            draw.text(
                box[:2],
                item_tag,
                font=fnt, fill=(255, 255, 255, 255))

        msg_img = self.bridge.cv2_to_imgmsg(np.array(img), "rgb8")
        self.pub_img.publish(msg_img)

        return detections

