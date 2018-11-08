project(conduit_load)
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0054 NEW)

set(COMPILER "gnu")
#set(COMPILER "clang")
#set(CLUSTER "catalyst")
set(CLUSTER "surface")
#set(CLUSTER "quartz")
#set(CLUSTER "sierra")

set(LBANN_DIR ../..)
set(LBANN_INSTALL_DIR ${LBANN_DIR}/build/${COMPILER}.Release.${CLUSTER}.llnl.gov/install)
include(${LBANN_DIR}/cmake/modules/SetupMPI.cmake)
include(${LBANN_DIR}/cmake/modules/SetupOpenMP.cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (NOT OpenMP_CXX_LIBRARIES)
  message("  Set OpenMP_CXX_LIBRARIES manually")
  set(OpenMP_CXX_LIBRARIES "-fopenmp")
endif()

set(CONDUIT_LOAD_EXE conduit_load)
set(CONDUIT_LOAD_SRCS conduit_load.cpp)
set(WITH_OPENCL OFF)

add_definitions(-Wall)
add_definitions(-O2)
add_definitions(-fopenmp)
add_definitions(-g)
add_definitions(-std=c++11)
add_definitions(-D_JAG_OFFLINE_TOOL_MODE_)
add_definitions(-DLBANN_HAS_OPENCV)
add_definitions(-DLBANN_HAS_CONDUIT)
add_definitions(-DLBANN_NO_OMP_FOR_DATA_READERS)


#list(APPEND OpenCV_DIR /usr/local/tools/opencv-3.0.0)
list(APPEND OpenCV_DIR ${LBANN_INSTALL_DIR}/../opencv/build)
list(APPEND OpenCV_DIR /usr)
find_package(OpenCV QUIET HINTS ${OpenCV_DIR})
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

if(NOT OpenCV_FOUND)
  set(OpenCV_DIR ${LBANN_INSTALL_DIR})
  set(OpenCV_LIBS "libopencv_highgui.so;libopencv_imgproc.so;libopencv_imgcodecs.so;libopencv_core.so")
  set(OpenCV_INCLUDE_DIRS "${OpenCV_DIR}/include")
  set(OpenCV_LIB_DIR "${OpenCV_DIR}/lib")
  message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
endif()

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})


#set(CONDUIT_DIR /p/lscratchh/brainusr/conduit/conduit-${CLUSTER})
set(CONDUIT_DIR /usr/workspace/wsb/icfsi/conduit/install-toss3-dev)
#set(CONDUIT_DIR /usr/workspace/wsb/icfsi/conduit/install-blueos-dev)
#set(HDF5_DIR /usr/workspace/wsb/icfsi/conduit-helpers/hdf5-${CLUSTER}/hdf5)
set(HDF5_DIR /usr/workspace/wsb/icfsi/conduit-helpers/hdf5-quartz/hdf5)

message(STATUS "CONDUIT_DIR: ${CONDUIT_DIR}")
set(CONDUIT_LIB_DIR "${CONDUIT_DIR}/lib")
set(CONDUIT_LIBRARIES "libconduit.so;libconduit_relay.so")
link_directories(${CONDUIT_LIB_DIR})
set(CONDUIT_INCLUDE_DIRS "${CONDUIT_DIR}/include")

if(HDF5_DIR)
  message(STATUS "HDF5_DIR: ${HDF5_DIR}")
  set(HDF5_LIB_DIR "${HDF5_DIR}/lib")
  set(HDF5_LIBRARIES "libhdf5.so")
  link_directories(${HDF5_LIB_DIR})
  set(HDF5_INCLUDE_DIRS "${HDF5_DIR}/include")
endif()

include_directories(SYSTEM ${CONDUIT_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})


find_package(MPI REQUIRED)
message(STATUS "Found MPI: ${MPI_CXX_COMPILER} ${MPI_C_COMPILER} ${MPI_Fortran_COMPILER}")
include_directories(${MPI_CXX_INCLUDE_PATH})


file(GLOB CONDUIT_LOAD_DEPEND_SRCS
     data_reader_jag_conduit.cpp
     lbann/utils/random.cpp
     ${LBANN_DIR}/src//utils/file_utils.cpp
     ${LBANN_DIR}/src/data_readers/image_utils.cpp
     ${LBANN_DIR}/src/data_readers/cv_utils.cpp
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

add_executable(${CONDUIT_LOAD_EXE} ${CONDUIT_LOAD_SRCS} ${CONDUIT_LOAD_DEPEND_SRCS})
target_link_libraries(${CONDUIT_LOAD_EXE} ${OpenCV_LIBS} ${CONDUIT_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
