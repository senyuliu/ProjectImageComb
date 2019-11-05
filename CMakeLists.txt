project(ImageCombine)

cmake_minimum_required(VERSION 3.5)

#######################OPENCV#############
FIND_PACKAGE(OpenCV REQUIRED)

add_compile_options(-fPIC)
add_compile_options(-std=c++11)
add_compile_options(-g)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_executable(ImageCombine StereoCalibration.cpp  shellCommand.h shellCommand.cpp ) 

target_link_libraries(ImageCombine ${OpenCV_LIBS} glog)
