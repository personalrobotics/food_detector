''' configurations for ada '''

import os


gpus = '0'
use_cuda = True

image_topic = '/camera/color/image_raw/compressed'
msg_type = 'compressed'
depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
depth_msg_type = 'raw'

camera_tf = 'camera_color_optical_frame'
camera_info_topic = '/camera/color/camera_info'

camera_to_table = 0.335
camera_tilt = 1e-8

num_classes = 6  # should match the trained model
checkpoint = os.path.join(
    os.path.dirname(__file__),
    '../../external/pytorch_retinanet/checkpoint/food_ckpt.pth')
label_map = os.path.join(
    os.path.dirname(__file__),
    '../../external/bite_selection_package/data/food_general_label_map.pbtxt')
spanet_checkpoint = os.path.join(
    os.path.dirname(__file__),
    '../../external/bite_selection_package/checkpoint/spanet_ckpt.pth')

pred_position = [0.5, 0.5]

node_name = 'food_spanet_detector'
marker_ns = 'food_item'
frequency = 20

# Used for wall detector
destination_frame = 'j2n6s200_link_base'

