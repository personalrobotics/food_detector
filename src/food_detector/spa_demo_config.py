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
    'checkpoint/food_ckpt.pth')#retinanet_spanet_all_ckpt.pth')
label_map = os.path.join(
    bsp_path,
    'data/food_label_map.pbtxt')
spanet_checkpoint = os.path.join(
    bsp_path,
    'checkpoint/spanet_ckpt.pth')#food_spanet_all_rgb_wall_wo_celery_carrot_bell_pepper_ckpt_best.pth')
spanet_wall_checkpoint = os.path.join(
    bsp_path,
    'checkpoint/food_spanet_all_rgb_wall_dr_ckpt_best.pth')#'checkpoint/spanet_with_wall_ckpt.pth')  # DEBUG: XM 

pred_position = [0.5, 0.5]

node_name = 'food_spanet_detector'
marker_ns = 'food_item'
frequency = 20

# Used for wall detector
destination_frame = 'map'

