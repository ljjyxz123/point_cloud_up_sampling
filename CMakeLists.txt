cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(point_cloud_up_sampling)

find_package(PCL 1.2 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (point_cloud_up_sampling stl2ply.cpp)
target_link_libraries (point_cloud_up_sampling ${PCL_LIBRARIES})
