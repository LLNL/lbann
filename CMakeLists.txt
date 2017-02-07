project(LBANN)
cmake_minimum_required(VERSION 2.8.12)
#set(LBANN_MAJOR_VERSION 0)
#set(LBANN_MINOR_VERSION 91)

# Check to see if we are in a git repo
execute_process(COMMAND git rev-parse --is-inside-work-tree OUTPUT_VARIABLE GIT_REPO OUTPUT_STRIP_TRAILING_WHITESPACE)

if(${GIT_REPO})
  # Get the git version so that we can embed it into the executable
  execute_process(COMMAND git --git-dir ${LBANN_HOME}/.git describe --abbrev=7 --dirty --always --tags OUTPUT_VARIABLE GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLBANN_VERSION=\\\"${GIT_VERSION}\\\"" )
else()
  if(LBANN_VER)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLBANN_VERSION=\\\"${LBANN_VER}\\\"" )
  else()
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLBANN_VERSION=\\\"v0.UKWN\\\"" )
  endif()
endif()

################################################################
# Options
################################################################

option(WITH_CUDA "Include Nvidia CUDA" ON)
option(WITH_CUDNN "Include Nvidia cuDNN" ON)
option(WITH_TBINF "Include Tensorboard interface" ON)
option(WITH_VTUNE "Link the Intel VTune profiling library" OFF)
option(VERBOSE "Verbose output" OFF)
if(NOT MAKE_NUM_PROCESSES)
  set(MAKE_NUM_PROCESSES 1)
endif()
if(NOT LIBRARY_TYPE)
  set(LIBRARY_TYPE SHARED)
endif()

################################################################
# Initialize build
################################################################

# Add CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Initialize C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")

# Disable all optimization in debug for better viewing under debuggers (cmake already adds -g)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# Special handling if we're compiling with Clang's address sanitizer
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CXX_VERSION)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   if(NOT (CXX_VERSION VERSION_GREATER 3.5 OR CXX_VERSION VERSION_EQUAL 3.5))
     message(FATAL_ERROR "LBANN & Elemental requires clang Version >= 3.5")
   endif()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address")
   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address")
endif()


# Set verbose output (should be 0 or 1)
if(VERBOSE)
  set(VERBOSE 1)
else()
  set(VERBOSE 0)
endif()

# Get installation directories
include(GNUInstallDirs)

################################################################
# Initialize RPATH (always full RPATH)
# Note: see https://cmake.org/Wiki/CMake_RPATH_handling
################################################################

# Use RPATH on OS X
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()

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
include(Elemental)
include(OpenCV)
include(protobuf)

# Optional dependencies
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
if(WITH_VTUNE)
  include(VTune)
endif()
include(Doxygen)

################################################################
# Check Compiler versions
################################################################

execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CXX_VERSION)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
   if(NOT (CXX_VERSION VERSION_GREATER 4.9 OR CXX_VERSION VERSION_EQUAL 4.9))
     message(FATAL_ERROR "LBANN & Elemental requires G++ Version >= 4.9")
   endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
   if(NOT (CXX_VERSION VERSION_GREATER 16.0 OR CXX_VERSION VERSION_EQUAL 16.0))
     message(FATAL_ERROR "LBANN & Elemental requires icpc Version >= 16.0")
   endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   if(NOT (CXX_VERSION VERSION_GREATER 3.5 OR CXX_VERSION VERSION_EQUAL 3.5))
     message(FATAL_ERROR "LBANN & Elemental requires clang Version >= 3.5")
   endif()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address")
else()
   message(FATAL_ERROR "Unsupported compiler: Unknown compiler vendor")
endif()

################################################################
# Build LBANN
################################################################

# Include LBANN header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Build nested source lists
macro(add_sources)
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
add_subdirectory(utility)

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
target_link_libraries(lbann ${PROTOBUF_LIBRARIES})

if(LBANN_HAS_CUDA)
  target_link_libraries(lbann ${CUDA_LIBRARIES})
endif()
if(LBANN_HAS_CUDNN)
  target_link_libraries(lbann ${cuDNN_LIBRARIES})
endif()
if(LBANN_HAS_VTUNE)
  target_link_libraries(lbann ${VTUNE_STATIC_LIB})
  target_link_libraries(lbann dl)
endif()
if(LBANN_HAS_TBINF)
  target_link_libraries(lbann TBinf)
endif()

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

################################################################
# Configuration summary
################################################################

message("== Configuration Summary ==")
message("  PROJECT_SOURCE_DIR:   ${PROJECT_SOURCE_DIR}")
message("  PROJECT_BINARY_DIR:   ${PROJECT_BINARY_DIR}")
message("  CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message("  CMAKE_BUILD_TYPE:     ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE MATCHES None)
  message("  CXX FLAGS:            ${CMAKE_CXX_FLAGS}")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
  message("  CXX FLAGS:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("  CXX FLAGS:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  message("  CXX FLAGS:            ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
message("  LBANN_HAS_MPI:        ${LBANN_HAS_MPI}")
message("  LBANN_HAS_ELEMENTAL:  ${LBANN_HAS_ELEMENTAL}")
message("  LBANN_HAS_OPENCV:     ${LBANN_HAS_OPENCV}")
message("  LBANN_HAS_CUDA:       ${LBANN_HAS_CUDA}")
message("  LBANN_HAS_CUDNN:      ${LBANN_HAS_CUDNN}")
message("  LBANN_HAS_PROTOBUF:   ${LBANN_HAS_PROTOBUF}")
message("  LBANN_HAS_TBINF:      ${LBANN_HAS_TBINF}")
message("  LBANN_HAS_VTUNE:      ${LBANN_HAS_VTUNE}")
message("  LBANN_HAS_DOXYGEN:    ${LBANN_HAS_DOXYGEN}")
