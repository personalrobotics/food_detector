cmake_minimum_required(VERSION 2.8.3)
project(food_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  conban_spanet
)

catkin_python_setup()

add_service_files(
  FILES
  GetAction.srv
  PublishLoss.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()
