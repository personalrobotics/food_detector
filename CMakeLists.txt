cmake_minimum_required(VERSION 3.0.2)
project(food_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  sensor_msgs
  message_generation
)

add_service_files(
  FILES
  SPANetTrigger.srv
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  visualization_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
