''' configurations for ada '''

from food_detector import FoodDetector

estimator = FoodDetector

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

num_classes = 17
checkpoint = 'external/pytorch-retinanet/checkpoint/food_ckpt.pth'
label_map = 'external/pytorch-retinanet/data/can_data/food_label_map.pkl'

spnet_checkpoint = 'external/bite_selection_package/checkpoints/spnet_ckpt.pth'

pred_position = [0.5, 0.5]

node_title = 'deep_pose'
marker_ns = 'food_item'
frequency = 20

