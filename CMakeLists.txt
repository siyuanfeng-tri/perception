cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(perception)
find_package(drake REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV 2.4 REQUIRED)
find_package(lcm REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

add_library(perception SHARED
  src/perception.cc
  src/point_cloud_fusion.cc
  src/jpg_over_lcm.cc)
target_link_libraries(perception
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  drake::drake)

install(TARGETS perception
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY include/perception
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
