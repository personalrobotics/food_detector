cmake_minimum_required(VERSION 2.8.3)
project(food_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_python_setup()
catkin_package()
