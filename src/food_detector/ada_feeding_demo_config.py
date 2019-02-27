''' configurations for ada '''

import os
import rospkg
import torch
rospack = rospkg.RosPack()
bsp_path = rospack.get_path('bite_selection_package')
pr_path = rospack.get_path('pytorch_retinanet')

gpus = '0'
use_cuda = torch.cuda.is_available()

image_topic = '/camera/color/image_raw/compressed'
msg_type = 'compressed'
depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
depth_msg_type = 'raw'

camera_tf = 'camera_color_optical_frame'
camera_info_topic = '/camera/color/camera_info'

camera_to_table = 0.335
camera_tilt = 1e-8

num_classes = 15  # should match the trained model
checkpoint = os.path.join(
    pr_path,
    'checkpoint/food_ckpt.pth')
label_map = os.path.join(
    bsp_path,
    'data/food_label_map.pbtxt')
spnet_checkpoint = os.path.join(
    bsp_path,
    'checkpoint/spnet_ckpt.pth')

pred_position = [0.5, 0.5]

marker_ns = 'food_item'
frequency = 20

# Used for wall detector
destination_frame = 'j2n6s200_link_base'

