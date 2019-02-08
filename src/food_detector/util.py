import torch
import os
from scipy.special import softmax
from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.retinanet_utils.encoder import DataEncoder

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

def load_label_map(label_map_file):
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


def get_index_of_class_name(label_map, class_name):
    for index, name in label_map.items():
        if name == class_name:
            return index
    return -1

def get_box_coordinates(box, img_shape):
    txmin = int(box[0] * img_shape[0])
    tymin = int(box[1] * img_shape[1])
    txmax = int(box[2] * img_shape[0])
    tymax = int(box[3] * img_shape[1])
    return txmin, tymin, txmax, tymax

def calculate_depth_from_depth_image(xmin, ymin, xmax, ymax, dimg):
    dimg_sliced = np.array(dimg)[int(xmin):int(xmax), int(ymin):int(ymax)]
    depth = dimg_sliced.flatten()
    depth = depth[depth > 0]
    if depth is None or len(depth) == 0:
        return -1
    z0 = np.mean(depth)
    return z0 / 1000.0  # mm to m

def calculate_depth(depth_img):
    depth = depth_img.flatten()
    depth = depth[depth > 0]
    depth = depth[abs(depth - np.mean(depth)) < np.std(depth)]
    if depth is None or len(depth) == 0:
        return -1
    z0 = np.mean(depth)
    return z0 / 1000.0  # mm to m

def add_group_item(grid, x, y, gidx):
    if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
        return False
    if grid[x, y] != 1:
        return False

    grid[x, y] = gidx
    return True

def propagate_group(grid, x, y, gidx):
    next_steps = [
        [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1],
        [x + 1, y + 1], [x - 1, y - 1], [x - 1, y + 1], [x + 1, y - 1]]
    for item in next_steps:
        if add_group_item(grid, item[0], item[1], gidx):
            return propagate_group(grid, item[0], item[1], gidx)

def group_rmask(rmask, rmask_prob):
    grid = np.ones_like(rmask)
    grid[rmask == -2] = 0

    gidx = 2
    for ri in range(grid.shape[0]):
        for ci in range(grid.shape[1]):
            if add_group_item(grid, ri, ci, gidx):
                grid = propagate_group(grid, ri, ci, gidx)
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