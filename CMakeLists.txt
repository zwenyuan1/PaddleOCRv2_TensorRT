cmake_minimum_required(VERSION 2.6)

project(OCR)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/local/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# project
include_directories(include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(rec src/Convert.cpp 
                src/det.cpp 
                src/rec.cpp 
                src/ocr.cpp 
                src/postprocess_op.cpp 
                src/preprocess_op.cpp 
                src/utility.cpp 
                main.cpp)
target_link_libraries(rec nvinfer nvinfer_plugin nvparsers nvonnxparser)
target_link_libraries(rec cudart)
target_link_libraries(rec ${OpenCV_LIBS})

add_definitions(-O2 -pthread)