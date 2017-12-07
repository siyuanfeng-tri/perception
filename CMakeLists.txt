cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(perception)
find_package(PCL REQUIRED)
find_package(OpenCV 2.4 REQUIRED)
find_package(drake REQUIRED)
find_package(lcm REQUIRED)

#include(ExternalProject)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

#ExternalProject_Add(rgbd_bridge
#  GIT_REPOSITORY https://github.com/siyuanfeng-tri/rgbd_bridge/
#  GIT_TAG 7dca978f41507ab53b1da446e385b2f6de2e5d6b)

#ExternalProject_Add(perception
#  DEPENDS rgbd_bridge
#  SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")

add_definitions(-std=c++1y -g -Wall -O2 -fPIC)

add_library(perception
  src/perception.cc
  src/point_cloud_fusion.cc
  src/jpg_over_lcm.cc)
target_link_libraries(perception
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  drake::drake)
