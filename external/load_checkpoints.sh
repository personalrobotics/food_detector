#!/bin/bash

if [[ $1 == "spnet" ]]; then
  echo "load spnet from mthrbrn"
  scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/GitHub/bite_selection_package/checkpoint ./bite_selection_package/
elif [[ $1 == "model1" ]]; then
  echo "load model1 from mthrbrn"
  scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/catkin_ws/src/deep_pose_estimators/src/deep_pose_estimators/external/laura_model1/checkpoint ./laura_model1
else
  echo "load retinanet from mthrbrn"
  scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/GitHub/pytorch-retinanet-foods/checkpoint ./pytorch_retinanet/
  scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/GitHub/pytorch-retinanet-foods/pretrained ./pytorch_retinanet/
fi

