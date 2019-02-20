#!/usr/bin/env python

import json
import numpy as np
import rospy
from tf import TransformListener

from deep_pose_estimators.pose_estimators import PoseEstimator
from deep_pose_estimators.detected_item import DetectedItem
from deep_pose_estimators.utils.ros_utils import get_transform_matrix
from deep_pose_estimators.utils import CameraSubscriber

from food_detector import ImagePublisher
from food_detector.util import load_retinanet, load_label_map

class RetinaNetDetector(PoseEstimator, CameraSubscriber, ImagePublisher):
    def __init__(self,
            retinanet_checkpoint,
            use_cuda,
            label_map_file,
            publisher_topic='food_detector',
            image_topic='/camera/color/image_raw/compressed',
            image_msg_type='compressed',
            depth_image_topic='/camera/aligned_depth_to_color/image_raw',
            camera_info_topic='/camera/color/camera_info',
            detection_frame = "camera_color_optical_frame",
            timeout=1.0):

        PoseEstimator.__init__(self)
        CameraSubscriber.__init__(self, image_topic, image_msg_type,
            image_compressed, depth_image_topic, point_cloud_topic,
            camera_info_topic)
        ImagePublisher.__init__(self, publisher_topic)

        self.retinanet, self.retinanet_transform, self.encoder =
            load_retinanet(use_cuda, retinanet_checkpoint)
        self.label_map = load_label_map(label_map_file)

        if self.retinanet is None:
            self.init_retinanet()

        if self.label_map is None:
            self.load_label_map()

        self.timeout = timeout

        self.selector_food_names = [
            "carrot", "melon", "apple", "banana", "strawberry"]
        self.selector_index = 0

    def create_detected_item(self, rvec, tvec, t_class_name, box_id,
                             db_key='food_item'):
        pose = quaternion_matrix(rvec)
        pose[:3, 3] = tvec

        return DetectedItem(
            frame_id=self.frame,
            marker_namespace='{}_{}'.format(self.title, t_class_name),
            marker_id=box_id,
            db_key=db_key,
            pose=pose,
            detected_time=rospy.Time.now())

    # TODO: SPNet changes this.
    def get_skewering_pose(self, txmin, txmax, tymin, tymax, width, height,
            img_msg, t_class_name):
        """
        @return skewering position and angle in the image.
        """
        center_x = (txmin + txmax) / 2.0
        center_y = (tymin + tymax) / 2.0
        return (center_x, center_y), 0.0

    def find_closest_box_and_update(self, x, y, class_name, tolerance=5):
        """
        Finds ths closest bounding box in the current list and
        updates it with the provided x, y
        @param x: Center x-position of a bounding box in 2D image
        @param y: Center y-position of a bounding box in 2D image
        @param class_name: Class name of the associated item
        @param tolerance: Pixel tolerance. If no box of same class is found
        within this tolerance, adds a new box with a new id
        @return Box id associated with the closest bounding box
        """
        min_distance = np.float('inf')
        matched_id = None
        largest_id = -1
        for bid, (bx, by) in self.detected_item_boxes[class_name].iteritems():
            distance = np.linalg.norm(np.array([x, y]) - np.array([bx, by]))
            largest_id = max(largest_id, bid)
            if distance >= tolerance:
                continue
            if distance < min_distance:
                min_distance = distance
                matched_id = bid
                matched_position = (bx, by)

        if matched_id is not None:
            self.detected_item_boxes[class_name][matched_id] = (x, y)
            print("Detected the closest box with id {} for {} at distance {}".format(matched_id, class_name, min_distance))
        else:
            self.detected_item_boxes[class_name][largest_id + 1] = (x, y)
            matched_id = largest_id + 1
            print("Adding a new box with id {} for {}".format(matched_id, class_name)

        return matched_id

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

    def calculate_depth(self, depth_img):
        depth = depth_img.flatten()
        depth = depth[depth > 0]
        depth = depth[abs(depth - np.mean(depth)) < np.std(depth)]
        if depth is None or len(depth) == 0:
            return -1
        z0 = np.mean(depth)
        return z0 / 1000.0  # mm to m

    def detect_objects(self):
        if self.img_msg is None:
            print('no input stream')
            return list()

        if self.depth_img_msg is None:
            print('no input depth stream')
            self.depth_img_msg = np.ones(self.img_msg.shape[:2])

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

            skewer_xy, skewer_angle = self.get_skewering_pose(
                    txmin, txmax, tymin, tymax, width, height,
                    copied_img_msg, t_class_name)

            class_box_id = self.find_closest_box_and_update(
                    center_x, center_y, t_class_name)

            cropped_depth = depth_img[
                int(max(tymin, 0)):int(min(tymax, height)),
                int(max(txmin, 0)):int(min(txmax, width))]
            z0 = self.calculate_depth(cropped_depth)
            if z0 < 0:
                print("skipping " + t_class_name + " due to invalid z0")
                continue

            if spBoxIdx >= 0:
                this_pos = skewer_xy
                this_ang = skewer_angle

                txoff = (txmax - txmin) * this_pos[0]
                tyoff = (tymax - tymin) * this_pos[1]
                pt = [txmin + txoff, tymin + tyoff]

                coff = 60
                cropped_depth = depth_img[
                    int(pt[1] - coff):int(pt[1] + coff),
                    int(pt[0] - coff):int(pt[1] + coff)]
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
                    rvec, tvec, t_class_name, t_class, class_box_id))

        self.visualize_detections(boxes, force_food, force_food_name)
        return detections


    def visualize_detections(self, boxes, force_food, force_food_name=""):
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

