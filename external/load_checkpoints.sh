#!/bin/bash

# Laura's model (baseline for spnet)
echo "load model1 from mthrbrn"
scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/catkin_ws/src/pose_estimators/src/pose_estimators/external/laura_model1/checkpoint ./laura_model1
