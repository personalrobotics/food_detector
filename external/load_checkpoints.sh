#!/bin/bash

echo "load spnet from mthrbrn"
mkdir -p ./bite_selection_package/checkpoint
scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/GitHub/bite_selection_package/checkpoint/spnet_ckpt.pth ./bite_selection_package/checkpoint/

echo "load retinanet from mthrbrn"
mkdir -p ./pytorch_retinanet/checkpoint
mkdir -p ./pytorch_retinanet/pretrained
scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/GitHub/pytorch-retinanet-foods/checkpoint/food_ckpt.pth ./pytorch_retinanet/checkpoint/
scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/GitHub/pytorch-retinanet-foods/pretrained/food_net.pth ./pytorch_retinanet/pretrained/

echo "load bite_selection_package data"
. ./bite_selection_package/data/load_data.sh

# Laura's model (baseline for spnet)
#  echo "load model1 from mthrbrn"
#  scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/catkin_ws/src/pose_estimators/src/pose_estimators/external/laura_model1/checkpoint ./laura_model1
