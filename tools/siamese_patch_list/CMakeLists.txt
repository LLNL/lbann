project(siamese_patches)
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)

set(COMPILER "gnu")
set(CLUSTER "catalyst")
#set(CLUSTER "surface")
#set(CLUSTER "quartz")
set(LBANN_DIR ../..)
set(LBANN_INSTALL_DIR ${LBANN_DIR}/build/${COMPILER}.Release.${CLUSTER}.llnl.gov/install)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SIAMESE_PATCHES_EXE siamese_patches)
set(SIAMESE_PATCHES_SRCS siamese_patches.cpp)
set(WITH_OPENCL OFF)

add_definitions(-Wall)
add_definitions(-O2)
add_definitions(-g)
add_definitions(-std=c++11)
add_definitions(-D_OFFLINE_PATCHES_NPZ_OFFLINE_TOOL_MODE_)


list(APPEND CNPY_DIR /usr)
find_package(CNPY QUIET HINTS ${CNPY_DIR})
message(STATUS "CNPY_DIR: ${CNPY_DIR}")

if(NOT CNPY_FOUND)
  set(CNPY_DIR ${LBANN_INSTALL_DIR})
  set(CNPY_LIBS "libcnpy.so;libz.so")
  set(CNPY_INCLUDE_DIRS "${CNPY_DIR}/include")
  set(CNPY_LIB_DIR "${CNPY_DIR}/lib")
  message(STATUS "CNPY_DIR: ${CNPY_DIR}")
endif()

include_directories(SYSTEM ${CNPY_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${CNPY_LIB_DIR})



file(GLOB SIAMESE_PATCHES_DEPEND_SRCS
     ${LBANN_DIR}/src/utils/file_utils.cpp
     ${LBANN_DIR}/src/utils/cnpy_utils.cpp
     ${LBANN_DIR}/src/data_readers/offline_patches_npz.cpp)

add_executable(${SIAMESE_PATCHES_EXE} ${SIAMESE_PATCHES_SRCS} ${SIAMESE_PATCHES_DEPEND_SRCS})
target_link_libraries(${SIAMESE_PATCHES_EXE} ${CNPY_LIBS})
