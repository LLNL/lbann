cmake_minimum_required(VERSION 3.9)

project(LBANN CXX)

# Add CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

#FIXME
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(LBANN_DEBUG TRUE)
endif ()

# Build with at least C++11 standard; allow newer standards.
if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD EQUAL 98)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif ()

# Convenience defines
string(TOUPPER "${PROJECT_NAME}" UPPER_PROJECT_NAME)
string(TOLOWER "${PROJECT_NAME}" LOWER_PROJECT_NAME)

# This will help define some targets later
if (CMAKE_VERSION VERSION_LESS 3.9)
  set(LBANN_PUBLIC_LINK_FLAG)
else ()
  set(LBANN_PUBLIC_LINK_FLAG "PUBLIC")
endif ()

#
# Version setup
#

# Check to see if we are in a git repo
execute_process(
  COMMAND git rev-parse --is-inside-work-tree
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_REPO
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if (GIT_REPO)
  # Get the git version so that we can embed it into the executable
  execute_process(
    COMMAND git --git-dir .git describe --abbrev=7 --dirty --always --tags
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${UPPER_PROJECT_NAME}_VERSION ${GIT_VERSION}
    CACHE STRING "LBANN's version string")
else ()
  set(${UPPER_PROJECT_NAME}_VERSION v0.UNKN
    CACHE STRING "LBANN's version string")
endif (GIT_REPO)

#
# Options
#

option(${UPPER_PROJECT_NAME}_WITH_CUDA "Include Nvidia CUDA" OFF)

option(${UPPER_PROJECT_NAME}_WITH_NCCL "Include Nvidia NCCL2" OFF)

option(${UPPER_PROJECT_NAME}_WITH_CUDNN "Include Nvidia cuDNN" ON)

option(${UPPER_PROJECT_NAME}_WITH_CNPY "Include cnpy" ON)

option(${UPPER_PROJECT_NAME}_WITH_TBINF "Include Tensorboard interface" ON)

option(${UPPER_PROJECT_NAME}_WITH_VTUNE
  "Link the Intel VTune profiling library" OFF)

option(${UPPER_PROJECT_NAME}_WITH_NVPROF
  "Enable NVTX-based instrumentation for nvprof" OFF)

option(${UPPER_PROJECT_NAME}_WITH_TOPO_AWARE
  "Enable topology-aware profiling (HWLOC)" ON)

option(${UPPER_PROJECT_NAME}_WITH_SOFTMAX_CUDA
  "Enable CUDA-based softmax layer" ON)

option(${UPPER_PROJECT_NAME}_VERBOSE "Enable verbose output" OFF)

option(${UPPER_PROJECT_NAME}_SEQUENTIAL_INITIALIZE
  "Sequentially consistent initialize" OFF)

# FIXME what if "fp16"?
set(${UPPER_PROJECT_NAME}_DATATYPE "float"
  CACHE STRING "The datatype to use in LBANN")

# Enable CUDA if necessary
if (${UPPER_PROJECT_NAME}_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # Handle C++11-ness of CUDA
    if (NOT CMAKE_CUDA_STANDARD OR CMAKE_CUDA_STANDARD EQUAL 98)
      set(CMAKE_CUDA_STANDARD 11)
      set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    endif ()

    # This enables all the CUDA-specific macros...
    find_package(CUDA)
  endif ()

  set(LBANN_HAS_CUDA ${CUDA_FOUND})

  if (LBANN_HAS_CUDA)
    set(LBANN_SOFTMAX_CUDA ${${UPPER_PROJECT_NAME}_WITH_SOFTMAX_CUDA})
  else ()
    message(STATUS "CUDA support not found. Disabling.")
    set(LBANN_SOFTMAX_CUDA FALSE)
    set(LBANN_WITH_SOFTMAX_CUDA OFF)
    set(LBANN_WITH_CUDA OFF)
  endif ()
endif ()

# Handle a few of the easy ones straight-away
set(LBANN_TOPO_AWARE ${${UPPER_PROJECT_NAME}_WITH_TOPO_AWARE})

# Enable parallel random matrix generation, if possible
if (${UPPER_PROJECT_NAME}_SEQUENTIAL_INITIALIZE)
  set(LBANN_SEQUENTIAL_CONSISTENCY TRUE)
  set(LBANN_PROCDET_DROPOUT TRUE)
  set(LBANN_PARALLEL_RANDOM_MATRICES FALSE)
else()
  set(LBANN_SEQUENTIAL_CONSISTENCY FALSE)
  set(LBANN_PROCDET_DROPOUT FALSE)
  set(LBANN_PARALLEL_RANDOM_MATRICES TRUE)
endif ()

#
# Initialize build
#

include(SetupCXX)

# Get installation directories
include(GNUInstallDirs)

################################################################
# Initialize RPATH (always full RPATH)
# Note: see https://cmake.org/Wiki/CMake_RPATH_handling
################################################################

# Use RPATH on OS X
if (APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif ()

# Use (i.e. don't skip) RPATH for build
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# Use same RPATH for build and install
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add build and install directories to RPATH
set(CMAKE_INSTALL_RPATH "${PROJECT_BINARY_DIR};${CMAKE_INSTALL_RPATH}")
set(CMAKE_INSTALL_RPATH "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR};${CMAKE_INSTALL_RPATH}")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_INSTALL_RPATH}")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR};${CMAKE_INSTALL_RPATH}")

################################################################
# Initialize dependencies
################################################################

# Required dependencies
include(SetupOpenMP)
include(SetupMPI)
include(SetupProtobuf)

# Setup the linear algebra library
include(SetupElemental)

# OpenCV installs a CMake configure file we can exploit
find_package(OpenCV REQUIRED)
set(LBANN_HAS_OPENCV ${OpenCV_FOUND})

# Setup some additional CUDA-y things
if (LBANN_HAS_CUDA)
  include(SetupCUDAToolkit)
  if (LBANN_WITH_NCCL2)
    include(SetupNCCL2)
  endif ()
  if (LBANN_WITH_CUDNN)
    include(SetupcuDNN)
  endif ()
endif ()

# Other optional dependencies
if (LBANN_WITH_TBINF)
  add_subdirectory(external/TBinf)
endif ()

if (LBANN_WITH_VTUNE)
  include(SetupVTune)
endif ()

if (LBANN_WITH_CNPY)
  find_package(CNPY)
  set(LBANN_HAS_CNPY ${CNPY_FOUND})
  if (NOT LBANN_HAS_CNPY)
    message(FATAL_ERROR
      "Requested LBANN_WITH_CNPY but CNPY not found. "
      "CNPY is now disabled. "
      "Try specifying CNPY_DIR as the root of a CNPY install. "
      "Alternatively, build with LBANN_WITH_CNPY=OFF.")
    set(LBANN_WITH_CNPY OFF)
  endif (NOT LBANN_HAS_CNPY)
endif (LBANN_WITH_CNPY)

if (LBANN_TOPO_AWARE)
  # LBANN also requires HWLOC. This seems sufficient for now.
  find_package(HWLOC)

  if (NOT HWLOC_FOUND)
    message(FATAL_ERROR
      "Requested LBANN_WITH_TOPO_AWARE but HWLOC not found. "
      "HWLOC is now disabled. "
      "Try specifying HWLOC_DIR as the root of an HWLOC install. "
      "Alternatively, build with LBANN_WITH_TOPO_AWARE=OFF.")
    set(LBANN_TOPO_AWARE OFF)
  endif (NOT HWLOC_FOUND)
endif (LBANN_TOPO_AWARE)

# Handle the documentation
add_subdirectory(doc)

################################################################
# Build LBANN
################################################################

# Write the configure file
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/configure_files/lbann_config.hpp.in"
  "${CMAKE_BINARY_DIR}/lbann_config.hpp"
  @ONLY)

# Add LBANN source files
include(LBANNCMakeUtilities)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(model_zoo)
add_subdirectory(model_zoo/tests)
add_subdirectory(model_zoo/historical)
add_subdirectory(tests)

# Create the LBANN library
add_library(lbann ${LBANN_SOURCES} ${LBANN_HEADERS} ${LBANN_CUDA_SOURCES})

# Use the IMPORTED targets when possible.
target_link_libraries(lbann PUBLIC LbannProto)
target_link_libraries(lbann PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(lbann PUBLIC MPI::MPI_CXX)
target_link_libraries(lbann PUBLIC protobuf::libprotobuf)
target_link_libraries(lbann PUBLIC ${HYDROGEN_LIBRARIES})
target_link_libraries(lbann PUBLIC ${OpenCV_LIBRARIES})

if (LBANN_HAS_TBINF)
  target_link_libraries(lbann PUBLIC TBinf)
endif ()

if (LBANN_HAS_CNPY)
  target_link_libraries(lbann PUBLIC CNPY::CNPY)
endif ()

if (LBANN_TOPO_AWARE)
  target_link_libraries(lbann PUBLIC HWLOC::hwloc)
endif ()


#FIXME HERE DOWN

if (LBANN_HAS_CUDA)
  target_link_libraries(lbann PUBLIC ${CUDA_LIBRARIES})
  if (WITH_NVPROF)
    add_definitions(-DLBANN_NVPROF)
    target_link_libraries(lbann PUBLIC ${NVTX_LIBRARIES})
  endif ()
  target_link_libraries(lbann PUBLIC ${cuBLAS_LIBRARIES})

  if(WITH_NCCL)
    target_link_libraries(lbann PUBLIC ${nccl2_LIBRARIES})
  endif()
endif()
if(LBANN_HAS_CUDNN)
  target_link_libraries(lbann PUBLIC ${cuDNN_LIBRARIES})
endif ()
if (LBANN_HAS_VTUNE)
  target_link_libraries(lbann PUBLIC ${VTUNE_STATIC_LIB})
  target_link_libraries(lbann PUBLIC dl)
endif ()


target_compile_options(lbann PUBLIC "-Werror")
target_include_directories(lbann PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)

################################################################
# Install LBANN
################################################################

# Install library
install(
  TARGETS lbann
  EXPORT lbann
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install header files
install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/lbann"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install contributor list, license, readme
install(
  FILES "${PROJECT_SOURCE_DIR}/CONTRIBUTORS"
        "${PROJECT_SOURCE_DIR}/LICENSE"
        "${PROJECT_SOURCE_DIR}/README.md"
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
)

# Display all of the accumulated compiler definitions
get_directory_property( DirDefs COMPILE_DEFINITIONS )

################################################################
# Configuration summary
################################################################

message("== Configuration Summary ==")
message("  PROJECT_SOURCE_DIR:   ${PROJECT_SOURCE_DIR}")
message("  PROJECT_BINARY_DIR:   ${PROJECT_BINARY_DIR}")
message("  CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message("  CMAKE_BUILD_TYPE:     ${CMAKE_BUILD_TYPE}")
if (CMAKE_BUILD_TYPE MATCHES None)
  message("  CXX FLAGS:            ${CMAKE_CXX_FLAGS}")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  message("  CXX FLAGS:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("  CXX FLAGS:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif (CMAKE_BUILD_TYPE MATCHES Debug)
  message("  CXX FLAGS:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif ()
message("  COMPILE_DEFINITIONS:  ${DirDefs}" )
message("  LBANN_HAS_MPI:        ${LBANN_HAS_MPI}")
message("  LBANN_HAS_ELEMENTAL:  ${LBANN_HAS_ELEMENTAL}")
message("  LBANN_HAS_OPENCV:     ${LBANN_HAS_OPENCV}")
message("  LBANN_HAS_CUDA:       ${LBANN_HAS_CUDA}")
message("  LBANN_HAS_CUDNN:      ${LBANN_HAS_CUDNN}")
message("  ELEMENTAL_USE_CUBLAS: ${ELEMENTAL_USE_CUBLAS}")
message("  LBANN_HAS_PROTOBUF:   ${LBANN_HAS_PROTOBUF}")
message("  LBANN_HAS_CNPY:       ${LBANN_HAS_CNPY}")
message("  LBANN_HAS_TBINF:      ${LBANN_HAS_TBINF}")
message("  LBANN_HAS_VTUNE:      ${LBANN_HAS_VTUNE}")
message("  LBANN_HAS_DOXYGEN:    ${LBANN_HAS_DOXYGEN}")
message("  LBANN_HAS_LBANN_PROTO:${LBANN_HAS_LBANN_PROTO}")
