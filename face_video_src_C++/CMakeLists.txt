cmake_minimum_required(VERSION 3.6)
project(faceDemo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

find_package(Caffe)
#message(FATAL_ERROR ${Caffe_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp MTCNN.cpp SimilarityTransform.cpp readPath.cpp video_face.cpp videoProcess.cpp CenterFace.cpp)
add_executable(face_samples ${SOURCE_FILES})

target_link_libraries(face_samples ${OpenCV_LIBS} )
target_link_libraries(face_samples ${Caffe_LIBRARIES})
