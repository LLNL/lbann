cmake_minimum_required(VERSION 3.9)

message("\nWelcome to the LBANN SuperBuild system.\n\n"
  "Please report issues on https://github.com/llnl/lbann/issues\n\n"
  "Good luck!\n\n")

project(LBANN_SuperBuild NONE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

option(LBANN_SB_BUILD_ALUMINUM "Pull and build Aluminum from Github" OFF)

option(LBANN_SB_BUILD_CNPY "Pull and build CNPY from Github" OFF)

option(LBANN_SB_BUILD_CUB "Pull CUB from Github" OFF)

option(LBANN_SB_BUILD_ELEMENTAL
  "Pull and build Elemental's main repository from Github" OFF)

option(LBANN_SB_BUILD_HYDROGEN "Pull and build Hydrogen from Github" OFF)

option(LBANN_SB_BUILD_JPEG_TURBO "Download and build JPEG turbo" OFF)

option(LBANN_SB_BUILD_OPENCV "Pull and build OpenCV from Github" OFF)

option(LBANN_SB_BUILD_PROTOBUF "Pull and build Google Protobuf from Github" OFF)

option(LBANN_SB_BUILD_LBANN "Pull and build LBANN from Github" OFF)

#
# Add the TPL subdirectories
#
include(LBANNSuperBuildCreateCMakeArguments)

if (LBANN_SB_BUILD_ALUMINUM)
  add_subdirectory(aluminum)
  list(APPEND _BUILD_PKGS Aluminum)
endif ()

if (LBANN_SB_BUILD_CNPY)
  add_subdirectory(cnpy)
  list(APPEND _BUILD_PKGS CNPY)
endif ()

if (LBANN_SB_BUILD_CUB)
  add_subdirectory(cub)
  list(APPEND _BUILD_PKGS cub)
endif ()

# Elemental is *SO* complicated.
if (LBANN_SB_BUILD_ELEMENTAL OR LBANN_SB_BUILD_HYDROGEN)
  option(LBANN_SB_BUILD_OPENBLAS "Pull and build OpenBLAS from Github" OFF)

  if (LBANN_SB_BUILD_OPENBLAS)
    add_subdirectory(openblas)
    list(APPEND _BUILD_PKGS OpenBLAS)
  endif ()

  if (LBANN_SB_BUILD_HYDROGEN)
    add_subdirectory(hydrogen)
    list(APPEND _BUILD_PKGS Hydrogen)
  else ()
    add_subdirectory(elemental)
    list(APPEND _BUILD_PKGS Elemental)
  endif ()
endif ()

if (LBANN_SB_BUILD_JPEG_TURBO)
  add_subdirectory(jpeg-turbo)
  list(APPEND _BUILD_PKGS jpeg-turbo)
endif ()

if (LBANN_SB_BUILD_OPENCV)
  add_subdirectory(opencv)
  list(APPEND _BUILD_PKGS OpenCV)
endif ()

if (LBANN_SB_BUILD_PROTOBUF)
  add_subdirectory(protobuf)
  list(APPEND _BUILD_PKGS Protobuf)
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
  string(TOUPPER "${pkg}" pkg_upper)
  if (NOT ${pkg_upper}_CMAKE_INSTALL_PREFIX)
    message("  -- ${pkg} (${${pkg}_CMAKE_INSTALL_PREFIX})")
  else()
    message("  -- ${pkg} (${${pkg_upper}_CMAKE_INSTALL_PREFIX})")
  endif ()
endforeach ()
message("\n-----------------------------------------------------------------\n")

# Add a custom target for bundling all things up
if (UNIX)
  find_program(__TAR_EXE tar)
  find_program(__FIND_EXE find)
  set(__WORKING_DIR "${CMAKE_BINARY_DIR}")
  if (__TAR_EXE AND __FIND_EXE)
    set(__cmd "${__FIND_EXE};.;\(;-ipath;*/stamp/*.log;-o;-ipath;*/CMakeFiles/CMake*.log;-o;-name;CMakeCache.txt;\);-exec;${__TAR_EXE};czf;all_output_logs.tar.gz;{};+")
    add_custom_target(gather-logs
      COMMAND "${__cmd}"
      BYPRODUCTS "${__WORKING_DIR}/all_output_logs.tar.gz"
      WORKING_DIRECTORY "${__WORKING_DIR}"
      COMMENT "Gathering all output logs."
      VERBATIM
      COMMAND_EXPAND_LISTS
      USES_TERMINAL)
  endif (__TAR_EXE AND __FIND_EXE)
endif (UNIX)
