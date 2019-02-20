import torch
import os
from scipy.special import softmax
from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.retinanet_utils.encoder import DataEncoder
import torchvision.transforms as transforms

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

