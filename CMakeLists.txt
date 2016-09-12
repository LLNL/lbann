project(LBANN)
cmake_minimum_required(VERSION 2.8.11)

################################################################
# Options
################################################################

option(WITH_CUDA "Include Nvidia CUDA" ON)
option(WITH_CUDNN "Include Nvidia cuDNN" ON)
option(WITH_TBINF "Include Tensorboard interface" ON)
option(VERBOSE "Verbose output" OFF)
if(NOT MAKE_NUM_PROCESSES)
  set(MAKE_NUM_PROCESSES 1)
endif()
if(NOT LIBRARY_TYPE)
  set(LIBRARY_TYPE SHARED)
endif()

################################################################
# Initialize RPATH (always full RPATH)
# Note: copied from https://cmake.org/Wiki/CMake_RPATH_handling
################################################################

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH FALSE)

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

# Set verbose output (should be 0 or 1)
if(VERBOSE)
  set(VERBOSE 1)
else()
  set(VERBOSE 0)
endif()

################################################################
# Initialize dependencies
################################################################

# Required dependencies
include(MPI)
include(Elemental)
include(OpenCV)

# Optional dependencies
include(OpenMP)
if(WITH_CUDA)
  include(CUDA)
endif()
if(CUDA_FOUND)
  include(CUB)
  if(WITH_CUDNN)
    include(cuDNN)
  endif()
endif()
if(WITH_TBINF)
  include(TBinf)
endif()

# Doxygen
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

# Build nested source lists
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

# Add LBANN source files
add_subdirectory(src)
add_subdirectory(model_zoo)

# Build LBANN source files
add_library(src OBJECT ${SRCS})

# Add dependencies
if(LBANN_BUILT_ELEMENTAL)
  add_dependencies(src project_Elemental)
endif()
if(LBANN_BUILT_OPENCV)
  add_dependencies(src project_OpenCV)
endif()
if(LBANN_BUILT_CUB)
  add_dependencies(src project_CUB)
endif()
if(LBANN_HAS_TBINF)
  add_dependencies(src TBinf)
endif()

# Build LBANN library
add_library(lbann ${LIBRARY_TYPE} $<TARGET_OBJECTS:src>)

# Link libraries
target_link_libraries(lbann ${MPI_CXX_LIBRARIES})
target_link_libraries(lbann ${Elemental_LIBRARIES})
target_link_libraries(lbann ${OpenCV_LIBRARIES})
if(LBANN_HAS_CUDA)
  target_link_libraries(lbann ${CUDA_LIBRARIES})
endif()
if(LBANN_HAS_CUDNN)
  target_link_libraries(lbann ${cuDNN_LIBRARIES})
endif()
if(LBANN_HAS_TBINF)
  target_link_libraries(lbann TBinf)
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
