project(LBANN)
cmake_minimum_required(VERSION 2.8.11)

################################################################
# Options
################################################################

option(FORCE_OPENCV_BUILD "Force a build of OpenCV" OFF)
option(FORCE_ELEMENTAL_BUILD "Force a build of Elemental" OFF)
if(NOT DEFINED MAKE_NUM_PROCESSES)
  set(MAKE_NUM_PROCESSES 1)
endif()

################################################################
# Initialize RPATH (always full RPATH)
# Note: copied from https://cmake.org/Wiki/CMake_RPATH_handling
################################################################

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

################################################################
# Initialize build
################################################################

# Add CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Get version ID using Git version
execute_process(COMMAND git describe --abbrev=7 --dirty --always --tags OUTPUT_VARIABLE GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVERSION=\\\"${GIT_VERSION}\\\"")

# Initialize C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")

################################################################
# Initialize dependencies
################################################################

# Required dependencies
include(MPI)
include(Elemental)
include(OpenCV)

# Optional dependencies
include(OpenMP)
include(CUDA)
if(CUDA_FOUND)
  include(cuDNN)
  include(CUB)
endif()
# include(protobuf)

#
# Build nested source lists
#
macro (add_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND SRCS "${_relPath}/${_src}")
        else()
            list (APPEND SRCS "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (SRCS ${SRCS} PARENT_SCOPE)
    endif()
endmacro()

# TBinf (Tensorboard interface)
# This must come first so the right protobuf libraries are found.
if(CMAKE_TBINF)
    include_directories(external/TBinf)
    add_sources(external/TBinf/TBinf.cpp)
    add_sources(external/TBinf/summary.pb.cc)
    add_sources(external/TBinf/event.pb.cc)
    set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D__HAVE_TBINF" )
    if(CMAKE_PROTOBUF)
        # Need to get ahead of the system protobuf version for now.
        include_directories(BEFORE "${CMAKE_PROTOBUF}/include")
        link_directories("${CMAKE_PROTOBUF}/lib")
    endif(CMAKE_PROTOBUF)
endif(CMAKE_TBINF)
macro(include_link_tbinf target)
    if(CMAKE_TBINF)
        target_link_libraries(${target} protobuf)
    endif(CMAKE_TBINF)
endmacro()


#
# Doxygen
#
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in 
               ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile @ONLY)
add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

################################################################
# Build LBANN
################################################################

# Include LBANN header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add LBANN source files
add_subdirectory(src)
add_subdirectory(model_zoo)

# Build LBANN source files
add_library(src OBJECT ${SRCS})
if(LBANN_BUILT_CUB)
  add_dependencies(src project_CUB)
endif()
if(LBANN_BUILT_OPENCV)
  add_dependencies(src project_OpenCV)
endif()
if(LBANN_BUILT_ELEMENTAL)
  add_dependencies(src project_Elemental)
endif()

# Build LBANN library
set(LIBRARY_TYPE SHARED)
add_library(lbann ${LIBRARY_TYPE} $<TARGET_OBJECTS:src>)

# Link libraries
target_link_libraries(lbann ${MPI_CXX_LIBRARIES})
target_link_libraries(lbann ${Elemental_LIBRARIES})
target_link_libraries(lbann ${OpenCV_LIBRARIES})
include_link_tbinf(lbann)
if(LBANN_HAS_CUDA)
  target_link_libraries(lbann ${CUDA_LIBRARIES})
endif()
if(LBANN_HAS_CUDNN)
  target_link_libraries(lbann ${cuDNN_LIBRARIES})
endif()

################################################################
# Configuration Summary
################################################################

message("== Configuration Summary ==")
message("  CMAKE_BUILD_TYPE:    ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES None)
  message("  CXX FLAGS:           ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
  message("  CXX FLAGS:           ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("  CXX FLAGS:           ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  message("  CXX FLAGS:           ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
message("  LBANN_HAS_MPI:       ${LBANN_HAS_MPI}")
message("  LBANN_HAS_ELEMENTAL: ${LBANN_HAS_ELEMENTAL}")
message("  LBANN_HAS_OPENCV:    ${LBANN_HAS_OPENCV}")
message("  LBANN_HAS_OPENMP:    ${LBANN_HAS_OPENMP}")
message("  LBANN_HAS_CUDA:      ${LBANN_HAS_CUDA}")
message("  LBANN_HAS_CUDNN:     ${LBANN_HAS_CUDNN}")
message("  LBANN_HAS_PROTOBUF:  ${LBANN_HAS_PROTOBUF}")
message("  LBANN_HAS_TBINF:     ${LBANN_HAS_TBINF}")
message("  DOXYGEN_FOUND:       ${DOXYGEN_FOUND}")
if(CMAKE_TBINF)
  message("   TBinf: ${CMAKE_TBINF}")
endif(CMAKE_TBINF)
