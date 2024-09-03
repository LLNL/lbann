################################################################################
## Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################
cmake_minimum_required(VERSION 3.22)
project(LBANN_SuperBuild NONE)

cmake_policy(SET CMP0097 NEW)

message("\nWelcome to the LBANN SuperBuild system.\n\nGood luck!\n")

if (PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not permitted.")
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(LBANNSuperBuildAddCMakeExternPkg)
include(LBANNSuperBuildAddPackages)
include(LBANNSuperBuildCreateCMakeArguments)
include(LBANNSuperBuildInitExternPkg)

# SuperBuild options
option(LBANN_SB_DEFAULT_CUDA_OPTS
  "Set Tom's preferred default options for CUDA builds."
  OFF)

option(LBANN_SB_DEFAULT_ROCM_OPTS
  "Set Tom's preferred default options for ROCm builds."
  OFF)

option(LBANN_SB_CLONE_VIA_SSH
  "Use SSH protocol instead of HTTPS for github." OFF)

set(LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY "COMMON"
  CACHE STRING "Package prefix selection strategy. [COMMON, PKG, PKG_LC].")

# List packages that we can build. The names are chosen first to align
# with variables specific to the package (e.g., Aluminum uses
# "ALUMINUM_" as its variable prefix, so we simplify our lives by
# using "ALUMINUM" as the package name). When the variable names do
# not include the package, we fall back on however the package refers
# to itself.
#
# NOTE: This list is ORDERED. "Second-order" dependencies come before
# "first-order" dependencies.
#
# FIXME: Address the above node so that the packages are known before
# descending into subdirectories.
lbann_sb_add_packages(
  # Ack, some "third-order" dependencies.
  hwloc
  NCCL
  RCCL

  # These are "second-order" dependencies
  adiak        # Caliper
  Caliper      # Aluminum, LBANN
  Aluminum     # Hydrogen, DiHydrogen
  Catch2       # Hydrogen, DiHydrogen
  HDF5         # Conduit
  JPEG-TURBO   # OpenCV
  OpenBLAS     # Hydrogen
  spdlog       # DiHydrogen, LBANN

  # These are the "first-order" dependencies
  cereal
  Clara
  CNPY
  Conduit
  hiptt
  Hydrogen     # DiHydrogen depends on H.
  DiHydrogen
  OpenCV
  protobuf
  zstr

  # This should just be provided on relevant systems.
  AWS_OFI_RCCL

  # And finally add the option to build LBANN
  LBANN)

# Note: This changes things around slightly from the original
# version. I have flattened this list to IGNORE the dependency
# graph. This makes it slightly more general-purpose (that is, I don't
# care WHY you want to build {HDF5, JPEG-turbo, OpenBLAS} but not
# {Conduit, OpenCV, Hydrogen}, so I'm just going to let you do that).

# Add the TPL subdirectories
foreach (pkg ${LBANN_SB_BUILD_PKGS})
  if (LBANN_SB_BUILD_${pkg})
    string(TOLOWER "${pkg}" pkg_lower)
    add_subdirectory(${pkg_lower})
  endif ()
endforeach ()

# Print a helpful(?) message
set(LBANN_SB_SUGG_CMAKE_PREFIX_PATH_TMP "${CMAKE_PREFIX_PATH}" "\$\{CMAKE_PREFIX_PATH\}")
message("\n-----------------------------------------------------------------\n")
message("LBANN SuperBuild will build the following packages:\n")
foreach (pkg ${LBANN_SB_BUILD_PKGS})
  if (${pkg}_CMAKE_INSTALL_PREFIX)
    message("  -- ${pkg} (${${pkg}_CMAKE_INSTALL_PREFIX})")
    list(PREPEND LBANN_SB_SUGG_CMAKE_PREFIX_PATH_TMP
      "${${pkg}_CMAKE_INSTALL_PREFIX}")
  elseif (LBANN_SB_${pkg}_PREFIX)
    message("  -- ${pkg} (${LBANN_SB_${pkg}_PREFIX})")
    list(PREPEND LBANN_SB_SUGG_CMAKE_PREFIX_PATH_TMP
      "${LBANN_SB_${pkg}_PREFIX}")
  else ()
    message("  -- ${pkg} (??????)")
  endif ()
endforeach ()
list(REMOVE_DUPLICATES LBANN_SB_SUGG_CMAKE_PREFIX_PATH_TMP)
string(REPLACE ";" ":" LBANN_SB_SUGG_CMAKE_PREFIX_PATH
  "${LBANN_SB_SUGG_CMAKE_PREFIX_PATH_TMP}")
message("\nIt may be useful to do the following:\n")
message("export CMAKE_PREFIX_PATH=${LBANN_SB_SUGG_CMAKE_PREFIX_PATH}\n")
message("or\n")
message("source lbann_sb_suggested_cmake_prefix_path.sh\n")
message("Note that these assume a Bourne-compatible shell.")
message("\n-----------------------------------------------------------------\n")
file(WRITE "${CMAKE_BINARY_DIR}/lbann_sb_suggested_cmake_prefix_path.sh"
  "export CMAKE_PREFIX_PATH=${LBANN_SB_SUGG_CMAKE_PREFIX_PATH}\n")
file(WRITE "${CMAKE_INSTALL_PREFIX}/logs/lbann_sb_suggested_cmake_prefix_path.sh"
  "export CMAKE_PREFIX_PATH=${LBANN_SB_SUGG_CMAKE_PREFIX_PATH}\n")

if (LBANN_SB_BUILD_AWS_OFI_RCCL)
  message("-----------------------------------------------------------------\n")
  message("You are building the AWS_OFI_RCCL plugin. This plugin's library")
  message("path *must* be in the dynamic library search path at run time to")
  message("have an effect. It may be useful to do the following:\n")
  message("export LD_LIBRARY_PATH=${LBANN_SB_AWS_OFI_RCCL_PREFIX}/lib:\$\{LD_LIBARY_PATH\}\n")
  message("-----------------------------------------------------------------\n")
  file(APPEND "${CMAKE_BINARY_DIR}/lbann_sb_suggested_cmake_prefix_path.sh"
    "export AWS_OFI_RCCL_LIBDIR=${LBANN_SB_AWS_OFI_RCCL_PREFIX}/lib\n
    export LD_LIBRARY_PATH=${LBANN_SB_AWS_OFI_RCCL_PREFIX}/lib:\$\{LD_LIBRARY_PATH\}\n")
endif ()

# Add a custom target for bundling all things up
if (UNIX)
  find_program(__FIND_EXE find)
  mark_as_advanced(__FIND_EXE)
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
