cmake_minimum_required(VERSION 3.6)

project(LBANN CXX)

# Build with at least C++11 standard; allow newer standards.
if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD EQUAL 98)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif ()

# Convenience defines
string(TOUPPER UPPER_PROJECT_NAME "${PROJECT_NAME}")
string(TOLOWER LOWER_PROJECT_NAME "${PROJECT_NAME}")

# This will help define some targets later
set(LBANN_PUBLIC_LINK_FLAG "PUBLIC")
if (CMAKE_VERSION VERSION_LESS 3.9)
  set(LBANN_PUBLIC_LINK_FLAG)
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

option(${UPPER_PROJECT_NAME}_WITH_CUDA "Include Nvidia CUDA" ON)

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

# Handle a few of the easy ones straight-away
set(LBANN_TOPO_AWARE ${${UPPER_PROJECT_NAME}_WITH_TOPO_AWARE})
set(LBANN_SOFTMAX_CUDA ${${UPPER_PROJECT_NAME}_WITH_SOFTMAX_CUDA})

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

# Temporary workaround to force CMake to recognize the XL
# compiler. The latest beta XL compiler is recognized as Clang.
if (CMAKE_CXX_COMPILER MATCHES "xlc" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_COMPILER_ID "XL")
endif ()

# Add CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
#FIXME
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Initialize C++ flags
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -fPIC -g -Wall -Wextra -Wno-unused-parameter -Wnon-virtual-dtor -Wshadow")

# Disable all optimization in debug for better viewing under debuggers (cmake already adds -g)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(LBANN_DEBUG TRUE)
endif ()


# Special handling if we're compiling with Clang's address sanitizer
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CXX_VERSION)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   if (NOT (CXX_VERSION VERSION_GREATER 3.5 OR CXX_VERSION VERSION_EQUAL 3.5))
     message(FATAL_ERROR "LBANN & Elemental requires clang Version >= 3.5")
   endif ()
   if (CMAKE_BUILD_TYPE MATCHES Debug)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address")
     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address")
   else()
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
   endif ()
endif ()

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
include(OpenMP)
include(MPI)
include(protobuf)
include(OpenCV)
include(LbannProto)

# Optional dependencies
if (WITH_CUDA)
  include(CUDA)
endif ()

if (ELEMENTAL_USE_CUBLAS)
  # Is variable EL_CUBLAS_FLAGS used?
  set(EL_CUBLAS_FLAGS "-DEL_USE_CUBLAS -I${CUDA_INCLUDE_DIRS}")
  add_definitions(-DEL_USE_CUBLAS)
endif ()

if (CUDA_FOUND)
  include(CUB)
  if (WITH_CUDNN)
    include(cuDNN)
  endif ()
  set(CUDA_LINK_LIBRARIES_KEYWORD ${LBANN_PUBLIC_LINK_FLAG})

  if(WITH_NCCL)
     add_definitions(-D__LIB_NCCL)
  endif()
endif()

include(Elemental)

if (WITH_TBINF)
  include(TBinf)
endif ()
if (WITH_VTUNE)
  include(VTune)
endif ()
if (WITH_CNPY)
  include(cnpy)
endif ()
include(Doxygen)

################################################################
# Check Compiler versions
################################################################

execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CXX_VERSION)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
   if (NOT (CXX_VERSION VERSION_GREATER 4.9 OR CXX_VERSION VERSION_EQUAL 4.9))
     message(FATAL_ERROR "LBANN & Elemental requires G++ Version >= 4.9")
   endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
   if (NOT (CXX_VERSION VERSION_GREATER 16.0 OR CXX_VERSION VERSION_EQUAL 16.0))
     message(FATAL_ERROR "LBANN & Elemental requires icpc Version >= 16.0")
   endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   if (NOT (CXX_VERSION VERSION_GREATER 3.5 OR CXX_VERSION VERSION_EQUAL 3.5))
     message(FATAL_ERROR "LBANN & Elemental requires clang Version >= 3.5")
   endif ()
   if (CMAKE_BUILD_TYPE MATCHES Debug)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address")
   else()
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
   endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "XL")
  # Version requirement for xlc++? The latest compiler on ray is 14.1.0
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.1.0)
    message(WARNING "LBANN & Elemental may not work with xlc++ Version < 14.1.0")
  endif ()
else()
   message(FATAL_ERROR "Unsupported compiler: Unknown compiler vendor (${CMAKE_CXX_COMPILER_ID})")
endif ()

################################################################
# Build LBANN
################################################################

# Include LBANN header files
include_directories(${PROJECT_SOURCE_DIR}/include)


# Build nested source lists
# CUDA sources need to be handled differently as they are not
# supported by the add_library command (native CUDA support is
# available starting from version 3.8, though).
macro(add_sources)
  file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  foreach (_src ${ARGN})
    get_filename_component(_ext ${_src} EXT)
    message(STATUS "EXT: ${_ext} (${_src})")
    if (_relPath)
      set(_src "${_relPath}/${_src}")
    endif ()
    if (_ext STREQUAL ".cu")
      list (APPEND CUDA_SRCS "${_src}")
    else()
      list (APPEND SRCS "${_src}")
    endif ()
  endforeach()
  if (_relPath)
    # propagate SRCS to parent directory
    set (SRCS ${SRCS} PARENT_SCOPE)
    set (CUDA_SRCS ${CUDA_SRCS} PARENT_SCOPE)
  endif ()
endmacro()

# Add LBANN source files
add_subdirectory(src)
add_subdirectory(model_zoo)
add_subdirectory(model_zoo/tests)
add_subdirectory(model_zoo/historical)
add_subdirectory(tests)

# Build LBANN source files
# Note that CUDA sources are not included as they are not
# supported. They are directly built and linked by the
# cuda_add_library(lbann) command.
add_library(src OBJECT ${SRCS})

# Add dependencies
if (LBANN_BUILT_ELEMENTAL)
  add_dependencies(src project_Elemental)
endif ()
if (LBANN_BUILT_OPENCV)
  add_dependencies(src project_OpenCV)
endif ()
if (LBANN_BUILT_CUB)
  add_dependencies(src project_CUB)
endif ()
if (LBANN_BUILT_CNPY)
  add_dependencies(src project_CNPY)
endif ()
if (LBANN_HAS_TBINF)
  add_dependencies(src TBinf)
endif ()
if (LBANN_BUILT_LBANN_PROTO)
  add_dependencies(src LbannProto)
endif ()
target_include_directories(src PUBLIC
  $<TARGET_PROPERTY:H::Hydrogen,INTERFACE_INCLUDE_DIRECTORIES>)

# Build LBANN library
# Use cuda_add_library when CUDA source files are included.
if (CUDA_FOUND)
  cuda_add_library(lbann ${LIBRARY_TYPE} $<TARGET_OBJECTS:src> ${CUDA_SRCS})
else()
  add_library(lbann ${LIBRARY_TYPE} $<TARGET_OBJECTS:src>)
endif ()

message("\n\nHYDROGEN_LIBRARIES = ${HYDROGEN_LIBRARIES}\n\n")
# Link libraries
target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${MPI_CXX_LIBRARIES})
target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${HYDROGEN_LIBRARIES})
target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${OpenCV_LIBRARIES})
target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${PROTOBUF_LIBRARIES})
target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${CNPY_LIBRARIES})

message("main CMakeFiles XXX LbannProto_LIBRARIES: ${LbannProto_LIBRARIES}")
# target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${LbannProto_LIBRARIES})

if (LBANN_HAS_CUDA)
  target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${CUDA_LIBRARIES})
  if (WITH_NVPROF)
    add_definitions(-DLBANN_NVPROF)
    target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${NVTX_LIBRARIES})
  endif ()
  target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${cuBLAS_LIBRARIES})

  if(WITH_NCCL)
    target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${nccl2_LIBRARIES})
  endif()
endif()
if(LBANN_HAS_CUDNN)
  target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${cuDNN_LIBRARIES})
endif ()
if (LBANN_HAS_VTUNE)
  target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${VTUNE_STATIC_LIB})
  target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} dl)
endif ()
if (LBANN_HAS_TBINF)
  target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} TBinf)
endif ()
if (LBANN_HAS_LBANN_PROTO)
  target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} LbannProto)
endif ()

if (WITH_TOPO_AWARE)
  # LBANN also requires HWLOC. This seems sufficient for now.
  find_package(HWLOC)

  if (HWLOC_FOUND)
    target_include_directories(src SYSTEM PUBLIC ${HWLOC_INCLUDE_PATH})
    target_link_libraries(lbann ${LBANN_PUBLIC_LINK_FLAG} ${HWLOC_LIBRARIES})
  else()
    message(FATAL_ERROR "Requested WITH_TOPO_AWARE but HWLOC not found. "
      "Try specifying HWLOC_DIR as the root of an HWLOC install. "
      "Alternatively, build with WITH_TOPO_AWARE=OFF.")
  endif ()
endif ()

target_compile_options(src PUBLIC "-Werror")
target_compile_options(lbann PUBLIC "-Werror")

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
