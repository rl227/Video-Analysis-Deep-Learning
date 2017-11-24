cmake_minimum_required(VERSION 2.8)
project(faceDemo)

set(CMAKE_PREFIX_PATH "/data_b/bd-recommend/zhenyutang/caffe/build")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

#find_package(Caffe)
#message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
#include_directories(${Caffe_INCLUDE_DIRS})
#add_definitions(${Caffe_DEFINITIONS})

include_directories(/data_b/bd-recommend/lzm/caffe/build/install/include
 /usr/local/cuda-8.0/targets/x86_64-linux/include)
set(SOURCE_FILES main.cpp MTCNN.cpp SimilarityTransform.cpp readPath.cpp image_face.cpp CenterFace.cpp)
add_executable(face_samples ${SOURCE_FILES})

target_link_libraries(face_samples ${OpenCV_LIBS} )
#target_link_libraries(face_samples ${Caffe_LIBRARIES})
target_link_libraries(face_samples /data_b/bd-recommend/lzm/caffe/build/lib/libcaffe.so
 /usr/local/lib/libboost_system.so
 /usr/local/lib/libglog.so)
