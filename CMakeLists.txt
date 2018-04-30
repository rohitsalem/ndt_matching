cmake_minimum_required(VERSION 2.8.3)
project(ndt)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
				roscpp
				sensor_msgs
				geometry_msgs
				pcl_conversions
				pcl_ros
				std_msgs)

## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(ndt src/ndt.cpp include/voxelGrid..h)
target_link_libraries(ndt ${catkin_LIBRARIES})
#add_dependencies(ndt)
