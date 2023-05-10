import torch
import os
from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.utils.encoder import DataEncoder
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
import json

from maskrcnn.model import load_model


def load_retinanet(use_cuda, checkpoint):
    retinanet = RetinaNet()
    if use_cuda:
        ckpt = torch.load(os.path.expanduser(checkpoint))
    else:
        ckpt = torch.load(
            os.path.expanduser(checkpoint), map_location='cpu')
    retinanet.load_state_dict(ckpt['net'])
    retinanet.eval()
    if use_cuda:
        retinanet = retinanet.cuda()

    print('Loaded RetinaNet from ' + checkpoint)

    retinanet_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    encoder = DataEncoder()

    return retinanet, retinanet_transform, encoder

def load_maskrcnn(use_cuda, checkpoint):
    maskrcnn = load_model(checkpoint, use_cuda=use_cuda)
    maskrcnn.eval()
    if use_cuda:
        maskrcnn = maskrcnn.cuda()

    print('Loaded MaskRCNN from ' + checkpoint)

    return maskrcnn 


def load_label_map(label_map):
    with open(os.path.expanduser(label_map), 'r') as f:
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

    return label_dict

def load_maskrcnn_label_map(label_map):
    with open(os.path.expanduser(label_map), 'r') as f:
        label_map = json.load(f)
    
    label_map = {int(k):v for k,v in label_map.items()}
    return label_map