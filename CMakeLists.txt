cmake_minimum_required(VERSION 3.8)

project(LBANN CXX)

# Prevent in-source builds
if (PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source build attempted; please clean the CMake cache and then "
    "switch to an out-of-source build, e.g.,\n"
    "rm -rf CMakeCache.txt CMakeFiles/\nmkdir build && "
    "cd build && cmake <options> ..\n")
endif ()

# Add CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
include(LBANNCMakeUtilities)

#FIXME
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(LBANN_DEBUG TRUE)
endif ()

if (NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
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

set(LBANN_VERSION_MAJOR 0)
set(LBANN_VERSION_MINOR 97)
set(LBANN_VERSION_PATCH 1)

set(LBANN_VERSION "${LBANN_VERSION_MAJOR}.${LBANN_VERSION_MINOR}.${LBANN_VERSION_PATCH}")

# Check to see if we are in a git repo
find_program(__GIT_EXECUTABLE git)
mark_as_advanced(__GIT_EXECUTABLE)
if (__GIT_EXECUTABLE)

  execute_process(
    COMMAND ${__GIT_EXECUTABLE} rev-parse --is-inside-work-tree
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE __BUILDING_FROM_GIT_SOURCES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (__BUILDING_FROM_GIT_SOURCES)
    # Get the git version so that we can embed it into the executable
    execute_process(
      COMMAND ${__GIT_EXECUTABLE} rev-parse --show-toplevel
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE __GIT_TOPLEVEL_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${__GIT_EXECUTABLE} rev-parse --git-dir
      WORKING_DIRECTORY "${__GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE __GIT_GIT_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${__GIT_EXECUTABLE} --git-dir "${__GIT_GIT_DIR}" describe
      --abbrev=7 --always --dirty --tags
      WORKING_DIRECTORY "${__GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE __GIT_DESCRIBE_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(LBANN_GIT_VERSION "${__GIT_DESCRIBE_VERSION}"
      CACHE STRING "LBANN's version string as told by git.")
  endif (__BUILDING_FROM_GIT_SOURCES)
endif (__GIT_EXECUTABLE)

if (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
  set(LBANN_GNU_LINUX TRUE)
endif ()

#
# Options
#

option(LBANN_WITH_ALUMINUM "Enable Aluminum all-reduce library" OFF)

option(LBANN_WITH_CNPY "Include cnpy" ON)

option(LBANN_WITH_CONDUIT "Enable Conduit library" OFF)

option(LBANN_WITH_CUDNN "Include Nvidia cuDNN" ON)

option(LBANN_WITH_HWLOC
  "Enable topology-aware optimizations" ON)

option(LBANN_WITH_NVPROF
  "Enable NVTX-based instrumentation for nvprof" OFF)

option(LBANN_WITH_TBINF "Include Tensorboard interface" ON)


option(LBANN_WITH_VTUNE
  "Link the Intel VTune profiling library" OFF)

# Enable parallel random matrix generation, if possible
option(LBANN_DETERMINISTIC
  "Use deterministic algorithms as much as possible." OFF)

option(LBANN_SEQUENTIAL_INITIALIZATION
  "Sequentially consistent initialization" OFF)

option(LBANN_DEBUG_PRINT_SUBTARGETS
  "Turn on debugging output of internal target properties." OFF)
mark_as_advanced(LBANN_DEBUG_PRINT_SUBTARGETS)

# This option is off by default because non-developers should not use
# this option under normal circumstances.
option(LBANN_WARNINGS_AS_ERRORS
  "Build with warnings promoted to errors." OFF)
mark_as_advanced(LBANN_WARNINGS_AS_ERRORS)

#
# The datatype option is not binary
#

# FIXME what if "fp16"?
set(LBANN_DATATYPE "float"
  CACHE STRING "The datatype to use in LBANN")

#
# Initialize build
#

# Get installation directories -- these get used in various places;
# best to just make them available
include(GNUInstallDirs)

include(SetupCXX)

################################################################
# Initialize dependencies
################################################################

# Required dependencies

# Setup the linear algebra library
find_package(Hydrogen 1.1.0 NO_MODULE
  HINTS ${Hydrogen_DIR} ${HYDROGEN_DIR} $ENV{Hydrogen_DIR} $ENV{HYDROGEN_DIR}
  PATH_SUFFIXES lib/cmake/hydrogen
  NO_DEFAULT_PATH)
if (NOT Hydrogen_FOUND)
  find_package(Hydrogen 1.1.0 NO_MODULE REQUIRED)
endif ()
set(LBANN_HAS_HYDROGEN ${Hydrogen_FOUND})

include(SetupOpenMP)
include(SetupMPI)
include(SetupProtobuf)

# OpenCV installs a CMake configure file we can exploit
find_package(OpenCV NO_MODULE
  HINTS ${OpenCV_DIR} ${OPENCV_DIR} $ENV{OpenCV_DIR} $ENV{OPENCV_DIR}
  PATH_SUFFIXES share/OpenCV
  NO_DEFAULT_PATH)
if (NOT OpenCV_FOUND)
  find_package(OpenCV NO_MODULE REQUIRED)
endif ()
set(LBANN_HAS_OPENCV ${OpenCV_FOUND})

# CUDA-ness of LBANN is 1:1 with Hydrogen. Iff Hydrogen has CUDA, LBANN gets CUDA.
set(LBANN_HAS_CUDA ${_HYDROGEN_HAVE_CUDA})
set(LBANN_WITH_CUDA ${LBANN_HAS_CUDA})

if (LBANN_HAS_CUDA)
  enable_language(CUDA)

  if (NOT CMAKE_CUDA_STANDARD OR CMAKE_CUDA_STANDARD EQUAL 98)
    set(CMAKE_CUDA_STANDARD 11)
  endif ()

  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif ()

if (LBANN_WITH_ALUMINUM)
  # Aluminum may have already been found by Hydrogen
  if (NOT Aluminum_FOUND)
    find_package(Aluminum NO_MODULE
      HINTS ${Aluminum_DIR} ${ALUMINUM_DIR} ${AL_DIR}
      $ENV{Aluminum_DIR} $ENV{ALUMINUM_DIR} $ENV{AL_DIR}
      PATH_SUFFIXES lib64/cmake/aluminum lib/cmake/aluminum
      NO_DEFAULT_PATH)
    if (NOT Aluminum_FOUND)
      find_package(Aluminum NO_MODULE)
    endif ()
  endif ()
  set(LBANN_HAS_ALUMINUM ${Aluminum_FOUND})

  # Report failure.
  if (NOT LBANN_HAS_ALUMINUM)
    message(WARNING
      "Requested LBANN_WITH_ALUMINUM but Aluminum not found. "
      "Aluminum is now disabled. "
      "Try specifying ALUMINUM_DIR as the root of an ALUMINUM install. "
      "Alternatively, build with LBANN_WITH_ALUMINUM=OFF to suppress "
      "this warning.")
    set(LBANN_WITH_ALUMINUM OFF)

  else ()

    if (AL_HAS_CUDA AND NOT LBANN_HAS_CUDA)
      message(WARNING
        "Aluminum has CUDA support but CUDA support has not been found.")
    endif ()

    option(LBANN_BUILT_WITH_SPECTRUM "LBANN was built with Spectrum MPI" OFF)
    if (LBANN_BUILT_WITH_SPECTRUM)
      set(LBANN_ALUMINUM_MPI_PASSTHROUGH ON)
    endif (LBANN_BUILT_WITH_SPECTRUM)

  endif (NOT LBANN_HAS_ALUMINUM)
endif (LBANN_WITH_ALUMINUM)

# Setup some additional CUDA-y things
if (LBANN_HAS_CUDA)
  if (NOT LBANN_WITH_CUDNN)
    message(WARNING
      "Despite the fact that it looks optional, cuDNN is currently required "
      "when building with CUDA support. You have tried LBANN_WITH_CUDNN=OFF. "
      "It will be searched for anyway.")
    set(LBANN_WITH_CUDNN ON)
  endif ()

  include(SetupCUDAToolkit)

  set(LBANN_HAS_CUDNN ${CUDNN_FOUND})

  if (LBANN_HAS_ALUMINUM AND AL_HAS_NCCL)
    set(LBANN_HAS_NCCL2 TRUE)
  else ()
    set(LBANN_HAS_NCCL2 FALSE)
  endif ()
endif (LBANN_HAS_CUDA)

# This shouldn't be here, but is ok for now. This will occasionally be
# part of another TPL's libraries (e.g., MKL), but it's no
# guarantee. There's no harm including it multiple times.
if (NOT DL_LIBRARY)
  find_library(DL_LIBRARY dl DOC "The dynamic loader library.")
  if (DL_LIBRARY)
    message(STATUS "Found dl: ${DL_LIBRARY}")
  else ()
    message(FATAL_ERROR
      "dl library not found! This is a required library.\n"
      "Please add the path to libdl to CMAKE_LIBRARY_PATH.")
  endif (DL_LIBRARY)
endif ()

# Other optional dependencies
if (LBANN_WITH_TBINF)
  add_subdirectory(external/TBinf)
endif ()

if (LBANN_WITH_VTUNE)
  find_package(VTune MODULE)

  if (VTune_FOUND)
    set(LBANN_VTUNE TRUE)
    set(LBANN_HAS_VTUNE TRUE)
  else ()
    set(LBANN_VTUNE FALSE)
    set(LBANN_HAS_VTUNE FALSE)
    set(LBANN_WITH_VTUNE OFF)
    message(WARNING
      "Requested LBANN_WITH_VTUNE=ON, but VTUNE support not detected. "
      "Support NOT enabled. "
      "Try setting VTUNE_DIR to point to the VTune install prefix "
      "and reconfigure.")
  endif (VTune_FOUND)
endif (LBANN_WITH_VTUNE)

if (LBANN_WITH_NVPROF)
  set(LBANN_NVPROF TRUE)
endif ()

if (LBANN_WITH_CNPY)
  find_package(CNPY REQUIRED)
  set(LBANN_HAS_CNPY ${CNPY_FOUND})
endif (LBANN_WITH_CNPY)

if (LBANN_WITH_HWLOC)
  find_package(HWLOC REQUIRED)
  set(LBANN_TOPO_AWARE ${HWLOC_FOUND})
endif (LBANN_WITH_HWLOC)

if (LBANN_WITH_CONDUIT)
  # Apparently we have to find HDF5, too.
  find_package(HDF5 CONFIG
    HINTS ${HDF5_DIR} $ENV{HDF5_DIR}
    PATH_SUFFIXES share/cmake/hdf5
    NO_DEFAULT_PATH)
  if (NOT CONDUIT_FOUND)
    find_package(HDF5 CONFIG)
  endif ()
  if (NOT CONDUIT_FOUND)
    find_package(HDF5 REQUIRED)
    set(HDF5_FOUND_WITH_MODULE TRUE)
  endif ()

  find_package(CONDUIT CONFIG
    NAMES conduit
    CONFIGS conduit.cmake
    HINTS ${CONDUIT_DIR} $ENV{CONDUIT_DIR}
    PATH_SUFFIXES lib64/cmake lib/cmake
    NO_DEFAULT_PATH)
  if (NOT CONDUIT_FOUND)
    find_package(CONDUIT CONFIG REQUIRED)
  endif ()

  # Ugh. I don't like that this requires intimate knowledge of
  # specific targets that CONDUIT exports. It should support
  # components.
  if (NOT TARGET conduit_relay_mpi)
    message(FATAL_ERROR "CONDUIT does not have proper MPI support.")
  endif ()

  if (NOT TARGET conduit OR NOT TARGET conduit_relay
      OR NOT TARGET conduit_blueprint)
    message(FATAL_ERROR "Missing some CONDUIT required library.")
  endif ()

  if (NOT TARGET conduit::conduit)
    add_library(conduit::conduit INTERFACE IMPORTED)
  endif ()

  set(_conduit_interface_link_libs
    "conduit;conduit_relay;conduit_relay_mpi;conduit_blueprint")
  if (HDF5_FOUND_WITH_MODULE)
    list(APPEND _conduit_interface_link_libs
      ${HDF5_LIBRARIES})

    set_target_properties(conduit::conduit
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS})
  endif ()

  set_target_properties(conduit::conduit
    PROPERTIES
    INTERFACE_LINK_LIBRARIES
    ${_conduit_interface_link_libs})

  set(CONDUIT_LIBRARIES conduit::conduit)
  set(LBANN_HAS_CONDUIT ${CONDUIT_FOUND})
endif (LBANN_WITH_CONDUIT)

# Handle the documentation
add_subdirectory(docs)

################################################################
# Build LBANN
################################################################

# Write the configure file
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/configure_files/lbann_config.hpp.in"
  "${CMAKE_BINARY_DIR}/lbann_config.hpp"
  @ONLY)

# Add LBANN source files
add_subdirectory(include)
add_subdirectory(src)

# Create the LBANN library
add_library(lbann ${LBANN_SOURCES} ${LBANN_HEADERS} ${LBANN_CUDA_SOURCES})

target_include_directories(lbann PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)

# Use the IMPORTED targets when possible.
target_link_libraries(lbann PUBLIC LbannProto)
target_link_libraries(lbann PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(lbann PUBLIC MPI::MPI_CXX)
target_link_libraries(lbann PUBLIC protobuf::libprotobuf)
target_link_libraries(lbann PUBLIC ${HYDROGEN_LIBRARIES})
target_link_libraries(lbann PUBLIC ${OpenCV_LIBRARIES})
target_link_libraries(lbann PUBLIC ${CONDUIT_LIBRARIES})

if (LBANN_HAS_TBINF)
  target_link_libraries(lbann PUBLIC TBinf)
endif ()

if (LBANN_HAS_CNPY)
  target_link_libraries(lbann PUBLIC CNPY::CNPY)
endif ()

if (LBANN_TOPO_AWARE)
  target_link_libraries(lbann PUBLIC HWLOC::hwloc)
endif ()

if (LBANN_HAS_ALUMINUM)
  target_link_libraries(lbann PUBLIC ${Aluminum_LIBRARIES})
endif ()

if (LBANN_HAS_VTUNE)
  target_link_libraries(lbann PUBLIC ${VTUNE_STATIC_LIB})
endif ()

if (TARGET LBANN_CXX_FLAGS_werror)
  target_link_libraries(lbann PUBLIC LBANN_CXX_FLAGS_werror)
endif ()

target_link_libraries(lbann PUBLIC ${DL_LIBRARY})

# Clean things up
include(LBANNDebugUtilities)
lbann_remove_default_include_paths_from_all_subtargets(lbann)
# This is to fix a bug with certain compilers interacting poorly with
# NVCC. In particular, the NVCC include line would have things like
# "-isystem=/usr/include" which would cause issues resolving STL
# include paths. Since compilers include "/usr/include" in their
# system include paths anyway (things searched by #include <...>), we
# can safely remove these from the explicit link line.

if (LBANN_DEBUG_PRINT_SUBTARGETS)
  lbann_print_all_subtargets(lbann)
endif ()

# Add the rest of the things
add_subdirectory(model_zoo)
add_subdirectory(model_zoo/tests)
add_subdirectory(model_zoo/jag_utils)
add_subdirectory(tests)
add_subdirectory(scripts)

################################################################
# Install LBANN
################################################################

include(CMakePackageConfigHelpers)

# Write the version file. This is independent of build/install tree.
write_basic_package_version_file(
  LBANNConfigVersion.cmake
  VERSION "${LBANN_VERSION}"
  COMPATIBILITY SameMajorVersion)

# This is for the build tree
set(INCLUDE_INSTALL_DIRS "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/include/lbann"
  "${CMAKE_BINARY_DIR}/include/lbann")
set(LIB_INSTALL_DIR "${CMAKE_BINARY_DIR}")
set(EXTRA_CMAKE_MODULE_DIR "${CMAKE_SOURCE_DIR}/cmake/modules")
configure_package_config_file(cmake/configure_files/LBANNConfig.cmake.in
  "${CMAKE_BINARY_DIR}/LBANNConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_BINARY_DIR}"
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Build tree export
export(EXPORT LBANNTargets NAMESPACE LBANN:: FILE LBANNTargets.cmake)

# Write the configure file for the install tree
set(INCLUDE_INSTALL_DIRS include)
set(LIB_INSTALL_DIR lib)
set(CMAKE_INSTALL_DIR lib/cmake/lbann)
set(EXTRA_CMAKE_MODULE_DIR)
configure_package_config_file(cmake/configure_files/LBANNConfig.cmake.in
  "${CMAKE_BINARY_DIR}/LBANNConfig.cmake.install"
  INSTALL_DESTINATION "${CMAKE_INSTALL_DIR}"
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Install library
install(
  TARGETS lbann
  EXPORT LBANNTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )

# Install export
install(EXPORT LBANNTargets
  NAMESPACE LBANN::
  DESTINATION "${CMAKE_INSTALL_DIR}")

# Install the cmake stuff
install(FILES
  "${PROJECT_BINARY_DIR}/LBANNConfig.cmake.install"
  DESTINATION "${CMAKE_INSTALL_DIR}"
  RENAME "LBANNConfig.cmake")
install(FILES "${PROJECT_BINARY_DIR}/LBANNConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DIR})
install(DIRECTORY cmake/modules
  DESTINATION "${CMAKE_INSTALL_DIR}"
  FILES_MATCHING PATTERN "*.cmake")

# Install header files
install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/lbann"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
install(
  FILES "${PROJECT_BINARY_DIR}/lbann_config.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Install contributor list, license, readme
install(
  FILES "${PROJECT_SOURCE_DIR}/CONTRIBUTORS"
        "${PROJECT_SOURCE_DIR}/LICENSE"
        "${PROJECT_SOURCE_DIR}/README.md"
  DESTINATION "${CMAKE_INSTALL_DOCDIR}")

################################################################
# Configuration summary
################################################################

# This creates a formatted string that contains a list of variables,
# one per line, with their values interpreted as TRUE or FALSE. The
# purpose is to provide uniform output, rather than an odd mixture of
# "1", "0", "ON", "OFF", "TRUE" and "FALSE".
macro(append_str_tf STRING_VAR)
  lbann_get_max_str_length(_max_length ${ARGN})
  math(EXPR _max_length "${_max_length} + 2")

  foreach(var ${ARGN})
    string(LENGTH "${var}" _var_length)
    math(EXPR _num_spaces "${_max_length} - ${_var_length}")
    lbann_get_space_string(_spaces ${_num_spaces})
    if (${var})
      string(APPEND ${STRING_VAR} "  ${var}:" "${_spaces}" "TRUE\n")
    else ()
      string(APPEND ${STRING_VAR} "  ${var}:" "${_spaces}" "FALSE\n")
    endif ()
  endforeach()
endmacro ()

# NOTE: message() outputs to stderr by default. We now use a string to
# maintain this information and then have cmake echo it to stdout. The
# only side effects are that if you use the CMake GUI, you won't see
# this output anymore (they only report stderr) and that if you add
# something to the list, you must remember your newline!
set(_str "\n== LBANN Configuration Summary ==\n\n")
string(APPEND _str "  PROJECT_SOURCE_DIR:   ${PROJECT_SOURCE_DIR}\n"
  "  PROJECT_BINARY_DIR:   ${PROJECT_BINARY_DIR}\n\n"
  "  CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}\n"
  "  CMAKE_BUILD_TYPE:     ${CMAKE_BUILD_TYPE}\n\n")
if (CMAKE_BUILD_TYPE MATCHES None)
  string(APPEND _str
    "  CXX FLAGS:            ${CMAKE_CXX_FLAGS}\n")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  string(APPEND _str
    "  CXX FLAGS:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}\n")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  string(APPEND _str
    "  CXX FLAGS:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}\n")
elseif (CMAKE_BUILD_TYPE MATCHES Debug)
  string(APPEND _str
    "  CXX FLAGS:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}\n")
endif ()
string(APPEND _str "\n")

#Print the true/false guys
append_str_tf(_str
  LBANN_GNU_LINUX
  LBANN_HAS_HYDROGEN
  LBANN_HAS_OPENCV
  LBANN_HAS_CUDA
  LBANN_HAS_CUDNN
  LBANN_HAS_NCCL2
  LBANN_HAS_PROTOBUF
  LBANN_HAS_CNPY
  LBANN_HAS_TBINF
  LBANN_HAS_VTUNE
  LBANN_NVPROF
  LBANN_HAS_DOXYGEN
  LBANN_HAS_LBANN_PROTO
  LBANN_HAS_ALUMINUM
  LBANN_HAS_CONDUIT)
string(APPEND _str
  "\n== End LBANN Configuration Summary ==\n")

# Output to stdout
execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${_str}")
set(_str)
