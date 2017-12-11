cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(perception)
find_package(drake CONFIG REQUIRED)
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
  bot2-core-lcmtypes::lcmtypes_bot2-core-cpp)

#get_property(DUDE TARGET drake::drake PROPERTY INTERFACE_LINK_LIBRARIES)
#message(STATUS ${DUDE})

install(TARGETS perception
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY include/perception
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
