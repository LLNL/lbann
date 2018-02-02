project(compute_mean)
cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0015 NEW)

set(COMPILER "gnu")
set(CLUSTER "surface")
set(LBANN_DIR ../..)
set(LBANN_BUILD_DIR ${LBANN_DIR}/build/${COMPILER}.${CLUSTER}.llnl.gov/install)
include(${LBANN_DIR}/cmake/modules/SetupMPI.cmake)
include(${LBANN_DIR}/cmake/modules/SetupOpenMP.cmake)
include_directories(SYSTEM ${LBANN_INCLUDE_DIRS})
include_directories(SYSTEM ${LBANN_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(COMPUTE_MEAN_EXE compute_mean)
set(COMPUTE_MEAN_SRCS compute_mean.cpp)
set(UNIFORM_MEAN_EXE uniform_mean)
set(UNIFORM_MEAN_SRCS uniform_mean.cpp)
set(WITH_OPENCL OFF)

add_definitions(-Wall)
add_definitions(-O2)
add_definitions(-g)
add_definitions(-std=c++11)
add_definitions(-DLBANN_HAS_OPENCV)


list(APPEND OpenCV_DIR /usr/local/tools/opencv-3.0.0)
list(APPEND OpenCV_DIR /usr)
find_package(OpenCV QUIET HINTS ${OpenCV_DIR})
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

if(NOT OpenCV_FOUND)
  set(OpenCV_DIR ${LBANN_BUILD_DIR})
  set(OpenCV_LIBS "libopencv_highgui.so;libopencv_imgproc.so;libopencv_imgcodecs.so;libopencv_core.so")
  set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
  set(OpenCV_LIB_DIR "${OpenCV_DIR}/lib")
  message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
endif()

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})


find_package(MPI REQUIRED)
message(STATUS "Found MPI: ${MPI_CXX_COMPILER} ${MPI_C_COMPILER} ${MPI_Fortran_COMPILER}")
include_directories(${MPI_CXX_INCLUDE_PATH})


file(GLOB COMPUTE_MEAN_DEPEND_SRCS
     mpi_states.cpp
     walltimes.cpp
     params.cpp
     image_list.cpp
     mean_image.cpp
     process_images.cpp
     lbann/utils/random.cpp
     lbann/utils/file_utils.cpp
     ${LBANN_DIR}/src/data_readers/cv_augmenter.cpp
     ${LBANN_DIR}/src/data_readers/cv_colorizer.cpp
     ${LBANN_DIR}/src/data_readers/cv_cropper.cpp
     ${LBANN_DIR}/src/data_readers/cv_decolorizer.cpp
     ${LBANN_DIR}/src/data_readers/cv_mean_extractor.cpp
     ${LBANN_DIR}/src/data_readers/cv_normalizer.cpp
     ${LBANN_DIR}/src/data_readers/cv_process.cpp
     ${LBANN_DIR}/src/data_readers/cv_process_patches.cpp
     ${LBANN_DIR}/src/data_readers/cv_subtractor.cpp
     ${LBANN_DIR}/src/data_readers/cv_transform.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks_patch_descriptor.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks_ROI.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks_stats.cpp)

file(GLOB UNIFORM_MEAN_DEPEND_SRCS
     lbann/utils/file_utils.cpp
     ${LBANN_DIR}/src/data_readers/cv_transform.cpp
     ${LBANN_DIR}/src/data_readers/cv_subtractor.cpp)

add_executable(${COMPUTE_MEAN_EXE} ${COMPUTE_MEAN_SRCS} ${COMPUTE_MEAN_DEPEND_SRCS})
target_link_libraries(${COMPUTE_MEAN_EXE} ${OpenCV_LIBS} ${Elemental_LIBS} ${MPI_CXX_LIBRARIES})
add_executable(${UNIFORM_MEAN_EXE} ${UNIFORM_MEAN_SRCS} ${UNIFORM_MEAN_DEPEND_SRCS})
target_link_libraries(${UNIFORM_MEAN_EXE} ${OpenCV_LIBS})
