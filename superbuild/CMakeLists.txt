cmake_minimum_required(VERSION 3.9)

message("\nWelcome to the LBANN SuperBuild system.\n\n"
  "This attempts to fill the shoes of a real package manager by building "
  "LBANN's dependencies, as well as LBANN itself.\n\n"
  "A few notes before we begin:\n"
  "  1. Read the README.md file.\n"
  "  2. Actually read the README.md file.\n"
  "  3. See 1 and 2.\n"
  "  4. You should probably use a real package manager instead (e.g., Spack)\n\n"
  "Good luck!\n\n")

project(LBANN_SuperBuild NONE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

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

if (LBANN_SB_BUILD_CNPY)
  add_subdirectory(cnpy)
endif ()

if (LBANN_SB_BUILD_CUB)
  add_subdirectory(cub)
endif ()

# Elemental is *SO* complicated.
if (LBANN_SB_BUILD_ELEMENTAL OR LBANN_SB_BUILD_HYDROGEN)
  option(LBANN_SB_BUILD_OPENBLAS "Pull and build OpenBLAS from Github" OFF)

  if (LBANN_SB_BUILD_OPENBLAS)
    add_subdirectory(openblas)
  endif ()

  if (LBANN_SB_BUILD_HYDROGEN)
    add_subdirectory(hydrogen)
  else ()
    add_subdirectory(elemental)
  endif ()
endif ()

if (LBANN_SB_BUILD_JPEG_TURBO)
  add_subdirectory(jpeg-turbo)
endif ()

if (LBANN_SB_BUILD_OPENCV)
  add_subdirectory(opencv)
endif ()

if (LBANN_SB_BUILD_PROTOBUF)
  add_subdirectory(protobuf)
endif ()

#
# Build LBANN using the above dependencies.
#

if (LBANN_SB_BUILD_LBANN)
  add_subdirectory(lbann)
endif ()
