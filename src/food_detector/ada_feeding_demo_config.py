''' configurations for ada '''

import os
import rospkg
import torch

rospack = rospkg.RosPack()
try:
	retinanet_base = rospack.get_path('pytorch_retinanet')
	biteselection_base = rospack.get_path('bite_selection_package')
except:
	print("WARNING: pytorch_retinanet and bite_selection_package not found")
	retinanet_base = ''
	biteselection_base = ''

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
checkpoint = os.path.join(retinanet_base, 'checkpoint/food_ckpt.pth')
label_map = os.path.join(biteselection_base, 'data/food_label_map.pbtxt')
spnet_checkpoint = os.path.join(biteselection_base, 'checkpoint/spnet_ckpt.pth')

pred_position = [0.5, 0.5]

marker_ns = 'food_item'
frequency = 20

# Used for wall detector
destination_frame = 'map'


# MAKSRCNN CONFIG
maskrcnn_base = rospack.get_path('maskrcnn')
maskrcnn_num_classes = 34
maskrcnn_checkpoint = os.path.join(maskrcnn_base, 'src/maskrcnn/best_train_model.pth')
maskrcnn_label_map = os.path.join(maskrcnn_base, 'data/json/id_to_food_mapping.json')

maskrcnn_image_topic = '/camera_1/color/image_raw'
maskrcnn_msg_type = 'raw'
maskrcnn_depth_image_topic = '/camera_1/aligned_depth_to_color/image_raw'
maskrcnn_depth_msg_type = 'raw'

maskrcnn_camera_tf = 'camera_color_optical_frame'
maskrcnn_camera_info_topic = '/camera_1/color/camera_info'

maskrcnn_destination_frame = 'map'    # possibly change??
maskrcnn_frequency = 15
maskrcnn_thresh = 0.4