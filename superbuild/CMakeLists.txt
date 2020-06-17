cmake_minimum_required(VERSION 3.9)

message("\nWelcome to the LBANN SuperBuild system.\n\n"
  "Please report issues on https://github.com/llnl/lbann/issues\n\n"
  "Good luck!\n")

# CXX is always required
project(LBANN_SuperBuild CXX)

if (PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "You have attempted an in-source build of LBANN.\n"
    "Please clean the CMake cache and switch to an out-of-source build.\n"
    "Example:\n"
    "rm -rf CMakeCache.txt CMakeFiles\n"
    "mkdir build && cd build\n"
    "cmake <lbann options> ..\n")
endif ()

get_filename_component(LBANN_SRC_DIR "${PROJECT_SOURCE_DIR}" PATH)
if (LBANN_SRC_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "You have attempted a superbuild in the LBANN source directory.\n"
    "Please clean the CMake cache and switch to an out-of-source build.\n"
    "Example:\n"
    "rm -rf CMakeCache.txt CMakeFiles\n"
    "mkdir build && cd build\n"
    "cmake <superbuild options> ../superbuild\n")
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

option(LBANN_SB_CLONE_VIA_SSH
  "Use SSH protocol instead of HTTPS for github." OFF)

option(LBANN_SB_BUILD_ALUMINUM "Pull and build Aluminum from Github" OFF)

option(LBANN_SB_BUILD_CATCH2 "Pull and install CATCH2 library from Github" OFF)

option(LBANN_SB_BUILD_CEREAL "Pull and install CEREAL library from Github" OFF)

option(LBANN_SB_BUILD_CLARA "Pull and install Clara library from Github" OFF)

option(LBANN_SB_BUILD_CNPY "Pull and build CNPY from Github" OFF)

option(LBANN_SB_BUILD_CONDUIT "Pull and build CONDUIT from Github" OFF)

option(LBANN_SB_BUILD_CUB "Pull CUB from Github" OFF)

option(LBANN_SB_BUILD_HDF5 "Pull and build HDF5 from the internet (not github)" OFF)

option(LBANN_SB_BUILD_HYDROGEN "Pull and build Hydrogen from Github" OFF)

option(LBANN_SB_BUILD_DIHYDROGEN "Pull and build DiHydrogen from Github" OFF)

option(LBANN_SB_BUILD_JPEG_TURBO "Download and build JPEG turbo" OFF)

option(LBANN_SB_BUILD_OPENCV "Pull and build OpenCV from Github" OFF)

option(LBANN_SB_BUILD_PROTOBUF "Pull and build Google Protobuf from Github" OFF)

option(LBANN_SB_BUILD_LBANN "Pull and build LBANN from Github" OFF)

#
# Add the TPL subdirectories
#
set(_GIT_REPOSITORY_TAG "GIT_REPOSITORY")
set(_GIT_TAG_TAG "GIT_TAG")
include(LBANNSuperBuildCreateCMakeArguments)

if (LBANN_SB_BUILD_ALUMINUM)
  add_subdirectory(aluminum)
  list(APPEND _BUILD_PKGS ALUMINUM)
endif ()

if (LBANN_SB_BUILD_CATCH2)
  add_subdirectory(catch2)
  list(APPEND _BUILD_PKGS CATCH2)
endif ()

if (LBANN_SB_BUILD_CLARA)
  add_subdirectory(clara)
  list(APPEND _BUILD_PKGS CLARA)
endif ()

if (LBANN_SB_BUILD_CEREAL)
  add_subdirectory(cereal)
  list(APPEND _BUILD_PKGS CEREAL)
endif ()

if (LBANN_SB_BUILD_CNPY)
  add_subdirectory(cnpy)
  list(APPEND _BUILD_PKGS CNPY)
endif ()

if (LBANN_SB_BUILD_CONDUIT)
  if (LBANN_SB_BUILD_HDF5)
    add_subdirectory(hdf5)
    list(APPEND _BUILD_PKGS HDF5)
  endif()

  add_subdirectory(conduit)
  list(APPEND _BUILD_PKGS CONDUIT)
endif ()

if (LBANN_SB_BUILD_CUB)
  add_subdirectory(cub)
  list(APPEND _BUILD_PKGS CUB)
endif ()

if (LBANN_SB_BUILD_HYDROGEN)
  option(LBANN_SB_BUILD_OPENBLAS "Pull and build OpenBLAS from Github" OFF)

  if (LBANN_SB_BUILD_OPENBLAS)
    add_subdirectory(openblas)
    list(APPEND _BUILD_PKGS OPENBLAS)
  endif ()

  add_subdirectory(hydrogen)
  list(APPEND _BUILD_PKGS HYDROGEN)
endif ()

if (LBANN_SB_BUILD_DIHYDROGEN)
  add_subdirectory(dihydrogen)
  list(APPEND _BUILD_PKGS DIHYDROGEN)
endif ()

if (LBANN_SB_BUILD_JPEG_TURBO)
  add_subdirectory(jpeg-turbo)
  list(APPEND _BUILD_PKGS JPEG-TURBO)
endif ()

if (LBANN_SB_BUILD_OPENCV)
  add_subdirectory(opencv)
  list(APPEND _BUILD_PKGS OPENCV)
endif ()

if (LBANN_SB_BUILD_PROTOBUF)
  add_subdirectory(protobuf)
  list(APPEND _BUILD_PKGS PROTOBUF)
endif ()

#
# Build LBANN using the above dependencies.
#

if (LBANN_SB_BUILD_LBANN)
  add_subdirectory(lbann)
  list(APPEND _BUILD_PKGS LBANN)
endif ()

message("\n-----------------------------------------------------------------\n")
message("LBANN SuperBuild will build the following packages:\n")
foreach (pkg ${_BUILD_PKGS})
  message("  -- ${pkg} (${${pkg}_CMAKE_INSTALL_PREFIX})")
endforeach ()
message("\n-----------------------------------------------------------------\n")

# Add a custom target for bundling all things up
if (UNIX)
  find_program(__FIND_EXE find)
  set(__WORKING_DIR "${CMAKE_BINARY_DIR}")
  if (__FIND_EXE)
    set(__cmd "${__FIND_EXE};.;\(;-ipath;*/stamp/*.log;-o;-ipath;*/CMakeFiles/CMake*.log;-o;-name;CMakeCache.txt;\);-exec;${CMAKE_COMMAND};-E;tar;czf;all_output_logs.tar.gz;--;{};+")
    add_custom_target(gather-logs
      COMMAND "${__cmd}"
      BYPRODUCTS "${__WORKING_DIR}/all_output_logs.tar.gz"
      WORKING_DIRECTORY "${__WORKING_DIR}"
      COMMENT "Gathering all output logs."
      VERBATIM
      COMMAND_EXPAND_LISTS
      USES_TERMINAL)

    add_custom_target(gather-all)
    add_dependencies(gather-all gather-logs)
    if (CMAKE_GENERATOR STREQUAL "Ninja")
      set(__cmd "${__FIND_EXE};.;-name;*.ninja;-exec;${CMAKE_COMMAND};-E;tar;czf;all_build_files.tar.gz;{};+")
    elseif (CMAKE_GENERATOR STREQUAL "Unix Makefiles")
      set(__cmd "${__FIND_EXE};.;\(;-name;link.txt;-o;-name;build.make;-o;-name;flags.make;\);-exec;${CMAKE_COMMAND};-E;tar;czf;all_build_files.tar.gz;{};+")
    else ()
      set(__cmd)
    endif ()
    if (__cmd)
      add_custom_target(gather-build
        COMMAND "${__cmd}"
        BYPRODUCTS "${__WORKING_DIR}/all_build_files.tar.gz"
        WORKING_DIRECTORY "${__WORKING_DIR}"
        COMMENT "Gathering all build files."
        VERBATIM
        COMMAND_EXPAND_LISTS
        USES_TERMINAL)
      add_dependencies(gather-all gather-build)
    endif ()
  endif (__FIND_EXE)
endif (UNIX)
