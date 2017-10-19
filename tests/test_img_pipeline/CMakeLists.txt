project(imgpipe)
cmake_minimum_required(VERSION 2.8)

set(LBANN_DIR ../..)
set(IMGPIPE_EXE imgpipe)
set(IMGPIPE_TEST_SRCS main.cpp)
set(WITH_OPENCL OFF)

add_definitions(-Wall)
add_definitions(-O2)
add_definitions(-g)
add_definitions(-std=c++11)
add_definitions(-D__LIB_OPENCV)


list(APPEND OpenCV_DIR /usr/local/tools/opencv-3.0.0)
list(APPEND OpenCV_DIR /usr)

find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR})
message(STATUS ${OpenCV_DIR})

set(Elemental_DIR /usr/workspace/wsb/yeom2/LBANN/El)
find_package(Elemental REQUIRED HINTS ${Elemental_DIR}/CMake/elemental)
message(STATUS ${Elemental_DIR})

include_directories(${OpenCV_INCLUDE_DIRS} ${LBANN_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${OpenCV_LIB_DIR})
#include_directories(${Elemental_DIR}/include)
include_directories(${Elemental_INCLUDE_DIRS})
link_directories(${Elemental_DIR}/lib)
link_directories(${Elemental_DIR}/lib64)
set(Elemental_LIBS ${Elemental_LIBRARIES})

file(GLOB IMGPIPE_SRCS ${LBANN_DIR}/src/utils/random.cpp
                          ${LBANN_DIR}/src/data_readers/image_utils.cpp
                          ${LBANN_DIR}/src/data_readers/cv_augmenter.cpp
                          ${LBANN_DIR}/src/data_readers/cv_colorizer.cpp
                          ${LBANN_DIR}/src/data_readers/cv_normalizer.cpp
                          ${LBANN_DIR}/src/data_readers/cv_process.cpp
                          ${LBANN_DIR}/src/data_readers/cv_process_patches.cpp
                          ${LBANN_DIR}/src/data_readers/cv_resizer.cpp
                          ${LBANN_DIR}/src/data_readers/cv_cropper.cpp
                          ${LBANN_DIR}/src/data_readers/cv_utils.cpp
                          ${LBANN_DIR}/src/data_readers/patchworks/patchworks.cpp
                          ${LBANN_DIR}/src/data_readers/patchworks/patchworks_patch_descriptor.cpp
                          ${LBANN_DIR}/src/data_readers/patchworks/patchworks_ROI.cpp
                          ${LBANN_DIR}/src/data_readers/patchworks/patchworks_stats.cpp)

add_executable(${IMGPIPE_EXE} ${IMGPIPE_SRCS} ${IMGPIPE_TEST_SRCS} )
target_link_libraries(${IMGPIPE_EXE} ${OpenCV_LIBS} ${Elemental_LIBS})
