project(patchworks)
cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0015 NEW)

set(COMPILER "gnu")
#set(CLUSTER "surface")
set(CLUSTER "catalyst")
set(LBANN_DIR ../..)
set(LBANN_BUILD_DIR ${LBANN_DIR}/build/${COMPILER}.Release.${CLUSTER}.llnl.gov/install)
include(${LBANN_DIR}/cmake/modules/SetupMPI.cmake)
include(${LBANN_DIR}/cmake/modules/SetupOpenMP.cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(LBANN_INCLUDE_DIR ${LBANN_BUILD_DIR}/include)
include_directories(${LBANN_INCLUDE_DIR})

set(PATCHWORKS_EXE patchworks)
set(PATCHWORKS_SRCS main.cpp patchworks_image.cpp patchworks_utils.cpp)


set(WITH_OPENCL OFF)

add_definitions(-Wall)
add_definitions(-O2)
add_definitions(-g)
add_definitions(-std=c++11)
add_definitions(-DLBANN_HAS_OPENCV)


#list(APPEND OpenCV_DIR /usr/local/tools/opencv-3.0.0)
#list(APPEND OpenCV_DIR /usr)
#find_package(OpenCV QUIET HINTS ${OpenCV_DIR})
#message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

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

list (APPEND Hydrogen_DIR ${LBANN_BUILD_DIR})
message(STATUS "Hydrogen_DIR: ${Hydrogen_DIR}")

include_directories(SYSTEM ${Hydrogen_INCLUDE_DIRS})
link_directories(${Hydrogen_DIR}/lib)
set(Hydrogen_LIBS "${Hydrogen_LIBRARIES};-lHydrogen;-lpmrrr;-lopenblas;-lpthread")

file(GLOB PATCHWORKS_DEPEND_SRCS
     lbann/utils/random.cpp
     ${LBANN_DIR}/src//utils/file_utils.cpp
     ${LBANN_DIR}/src/data_readers/image_utils.cpp
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
     ${LBANN_DIR}/src/data_readers/cv_utils.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks_patch_descriptor.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks_ROI.cpp
     ${LBANN_DIR}/src/data_readers/patchworks/patchworks_stats.cpp)

link_directories(${LBANN_DIR}/lib64)
link_directories(${LBANN_DIR}/lib)
add_executable(${PATCHWORKS_EXE} ${PATCHWORKS_DEPEND_SRCS} ${PATCHWORKS_SRCS})
target_link_libraries(${PATCHWORKS_EXE} ${OpenCV_LIBS} ${Hydrogen_LIBS} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} ${OpenMP_Fortran_LIBRARIES})
