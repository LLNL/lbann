project (LBANN)
cmake_minimum_required (VERSION 2.8.8)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMakeModules)

#
# Options
#
#option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Use Rpath for dependencies." TRUE)
#option(HAVOQGT_BUILD_TEST "Build HAVOQGT tests." TRUE)

################################################################################
# Always full RPATH
################################################################################
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
################################################################################

# Get the git version so that we can embed it into the executable
execute_process(COMMAND git describe --abbrev=7 --dirty --always --tags OUTPUT_VARIABLE GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVERSION=\\\"${GIT_VERSION}\\\"" )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )

#
# Main Include directory
#
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/include/lbann")
include_directories("${PROJECT_SOURCE_DIR}")

include(CheckSymbolExists)

#
# Link in OpenMP
#
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#
# Link in CUDA,cuDNN
#
find_package(CUDA QUIET)
if (CUDA_FOUND AND CMAKE_CUDNN_DIR)
    include(FindCUDA)
    include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    include_directories("${CMAKE_CUDNN_DIR}/include")
    link_directories("${CMAKE_CUDNN_DIR}/lib64")
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_CUDNN" )
    #CUDA_LIBS = -lcudart -lcublas
    set(CUDNN_LIBRARIES -lcudnn)
endif()

macro(link_cudnn target)
 target_link_libraries(${target} ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES})
endmacro()

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

# Link in OpenCV
include_directories("/usr/gapps/brain/installs/generic/include")
link_directories("/usr/gapps/brain/installs/generic/lib")
#CV_LIBS = -lopencv_core -lopencv_highgui
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D__LIB_OPENCV" )

#
# MPI is required
#
find_package( MPI REQUIRED )
macro(include_link_mpi target)
  include_directories(${MPI_INCLUDE_PATH})
  target_link_libraries(${target} ${MPI_LIBRARIES})
endmacro()

#
# OpenCV is required
#
#set(OpenCV_STATIC ON)
#set(OpenCV_CUDA OFF)

find_package( OpenCV REQUIRED HINTS ${OpenCV_DIR})
#find_package( OpenCV REQUIRED HINTS ${CMAKE_OPENCV_DIR})
if(OpenCV_FOUND)
  include_directories("${OpenCV_INCLUDE_DIRS}")
  set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D__LIB_OPENCV" )
  message("-- Found OpenCV version: ${OpenCV_VERSION} ${OpenCV_DIR}")
endif()

macro(link_opencv target)
 target_link_libraries(${target} ${OpenCV_LIBRARIES})
endmacro()

if(NOT ${CMAKE_CXX_COMPILER} MATCHES "icpc")
  find_package( OpenBLAS HINTS ${OpenBLAS_DIR})
  if(OpenBLAS_FOUND)
    include_directories("${OpenBLAS_INCLUDE_DIRS}")
    message("-- Found OpenBLAS version: ${OpenBLAS_VERSION} ${OpenBLAS_DIR}")
  endif()
endif()

macro(link_openblas target)
 target_link_libraries(${target} ${OpenBLAS_LIBRARIES})
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

#
# Elemental is required
#

if(CMAKE_ELEMENTAL_DIR)
   include_directories("${CMAKE_ELEMENTAL_DIR}/include/")
   link_directories("${CMAKE_ELEMENTAL_DIR}/lib/")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_ELEMENTAL" )
   message("-- Manually including Elemental Libraries: ${CMAKE_ELEMENTAL_DIR}")
else()
find_package(Elemental REQUIRED HINTS ${Elemental_ROOT_DIR})
if(Elemental_FOUND)
   set(CMAKE_CXX_FLAGS "${Elemental_COMPILE_FLAGS}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__LIB_ELEMENTAL" )
   include_directories(${Elemental_INCLUDE_DIRS})
   add_definitions(${Elemental_DEFINITIONS})
   message("-- Found Elemental Libraries: ${Elemental_LIBRARIES} ${Elemental_DIR}")
else()
#     add_subdirectory(external/Elemental)
   message(FATAL_ERROR "Could not locate Elemental installation")
endif()

macro(link_elemental target)
    target_link_libraries(${target} ${Elemental_LIBRARIES})
    if(Elemental_LINK_FLAGS)
       set_target_properties(${target} PROPERTIES LINK_FLAGS ${Elemental_LINK_FLAGS})
    endif()
endmacro()
endif()

include_directories(${MPI_CXX_INCLUDE_PATH})


# Build LBANN

add_subdirectory (src)
add_subdirectory (model_zoo)
#enable_testing()
#add_subdirectory (test)

add_library(src OBJECT ${SRCS})
if(NOT CMAKE_ELEMENTAL_DIR)
    add_dependencies(src El)
endif(NOT CMAKE_ELEMENTAL_DIR)

set( LIBRARY_TYPE SHARED )
add_library(lbann ${LIBRARY_TYPE} $<TARGET_OBJECTS:src>)

# Link in found libraries
link_opencv(lbann)
  include_link_tbinf(lbann)
  if(CMAKE_ELEMENTAL_DIR)
  target_link_libraries(lbann El)
else()
  link_elemental(lbann)
endif()

if(OpenBLAS_FOUND)
  link_openblas(lbann)
endif()

if(CUDA_FOUND)
  link_cudnn(lbann)
endif()

#link_directories(core)
#
# Configuration Summary
#
message("== Configuration Summary ==")
message("   CMAKE_BUILD_TYPE:   ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES None)
  message("        - CXX FLAGS:   ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
  message("        - CXX FLAGS:   ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("        - CXX FLAGS:   ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  message("        - CXX FLAGS:   ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
#message("   LBANN_BUILD_TEST:   ${LBANN_BUILD_TEST}")
message("   MPI_FOUND:             ${MPI_FOUND}")
message("   DOXYGEN_FOUND:   ${DOXYGEN_FOUND}")
message("   Elemental_FOUND:   ${Elemental_FOUND}")
message("   OpenCV_FOUND:      ${OpenCV_FOUND}")
#MPI params found
if(MPI_FOUND)
  message("   MPIEXEC: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} PROCS 
          ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
endif(MPI_FOUND)
if(CMAKE_TBINF)
  message("   TBinf: ${CMAKE_TBINF}")
endif(CMAKE_TBINF)
