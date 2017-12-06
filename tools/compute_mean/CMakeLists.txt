project(compute_mean)
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)

set(CLUSTER "surface")
#set(CLUSTER "quartz")
set(LBANN_DIR ../..)
set(LBANN_BUILD_DIR ${LBANN_DIR}/build/${CLUSTER}.llnl.gov)
include(${LBANN_DIR}/cmake/MPI.cmake)

set(COMPUTE_MEAN_EXE compute_mean)
set(COMPUTE_MEAN_TEST_SRCS compute_mean.cpp)
set(WITH_OPENCL OFF)

add_definitions(-Wall)
add_definitions(-O2)
add_definitions(-g)
add_definitions(-std=c++11)
add_definitions(-D__LIB_OPENCV)


list(APPEND OpenCV_DIR /usr/local/tools/opencv-3.0.0)
list(APPEND OpenCV_DIR /usr)
find_package(OpenCV QUIET HINTS ${OpenCV_DIR})
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

if(NOT OpenCV_FOUND)
  set(OpenCV_DIR ${LBANN_BUILD_DIR})
  set(OpenCV_LIBS "libopencv_core.so;libopencv_highgui.so;libopencv_imgcodecs.so;libopencv_imgproc.so")
  set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
  set(OpenCV_LIB_DIR "${OpenCV_DIR}/lib")
  message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
endif()

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS} ${LBANN_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${OpenCV_LIB_DIR})

list (APPEND Elemental_DIR ${LBANN_BUILD_DIR})
find_package(Elemental REQUIRED HINTS ${Elemental_DIR}/CMake/elemental)
message(STATUS "Elemental_DIR: ${Elemental_DIR}")

#Elemental library is required just becase of the dependency of utils/random.hpp is included in cv_cropper.hpp
#include_directories(SYSTEM ${Elemental_DIR}/include)
include_directories(SYSTEM ${Elemental_INCLUDE_DIRS})
link_directories(${Elemental_DIR}/lib)
link_directories(${Elemental_DIR}/lib64)
set(Elemental_LIBS ${Elemental_LIBRARIES})


find_package(MPI REQUIRED)
message(STATUS "Found MPI: ${MPI_CXX_COMPILER} ${MPI_C_COMPILER} ${MPI_Fortran_COMPILER}")
include_directories(${MPI_CXX_INCLUDE_PATH})


file(GLOB COMPUTE_MEAN_SRCS
     file_utils.cpp
     mpi_states.cpp
     walltimes.cpp
     params.cpp
     image_list.cpp
     mean_image.cpp
     process_images.cpp
     ${LBANN_DIR}/src/utils/random.cpp
     ${LBANN_DIR}/src/data_readers/cv_augmenter.cpp
     ${LBANN_DIR}/src/data_readers/cv_colorizer.cpp
     ${LBANN_DIR}/src/data_readers/cv_normalizer.cpp
     ${LBANN_DIR}/src/data_readers/cv_process.cpp
     ${LBANN_DIR}/src/data_readers/cv_process_patches.cpp
     ${LBANN_DIR}/src/data_readers/cv_cropper.cpp
     ${LBANN_DIR}/src/data_readers/cv_mean_extractor.cpp
     ${LBANN_DIR}/src/data_readers/cv_utils.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks_patch_descriptor.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks_ROI.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks_stats.cpp)

add_executable(${COMPUTE_MEAN_EXE} ${COMPUTE_MEAN_SRCS} ${COMPUTE_MEAN_TEST_SRCS} )
target_link_libraries(${COMPUTE_MEAN_EXE} ${OpenCV_LIBS} ${Elemental_LIBS} ${MPI_CXX_LIBRARIES})
