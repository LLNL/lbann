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
option(LBANN_SB_LBANN_ENABLE_EXTRA_UTILS
  "Enable the utility targets in LBANN"
  OFF)

option(LBANN_SB_FWD_LBANN_LBANN_WARNINGS_AS_ERRORS
  "Build with -Werror"
  OFF)

option(LBANN_SB_FWD_LBANN_LBANN_WITH_CALIPER
  "Build LBANN with Caliper support"
  ${LBANN_SB_BUILD_Caliper})

lbann_sb_add_cmake_extern_pkg(
  NAME LBANN
  LANGUAGES C CXX
  OPTIONAL_LANGUAGES CUDA HIP
  GITHUB_URL LLNL/lbann.git
  GIT_TAG "develop"
  DEPENDS_ON
  Aluminum DiHydrogen Hydrogen
  Caliper
  Catch2
  Clara
  CNPY
  Conduit HDF5
  hipTT
  OpenCV JPEG-TURBO
  OpenBLAS
  cereal
  protobuf
  spdlog
  zstr
)

if (LBANN_SB_LBANN_ENABLE_EXTRA_UTILS)

  # Ensure the JAG utils are built
  ExternalProject_Add_Step(LBANN build-jag-utils
    COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> --target jag-utils
    COMMENT "Performing building of JAG utils for 'LBANN'"
    DEPENDEES build
    DEPENDERS install
    LOG 1
    USES_TERMINAL 1)

  # Ensure the ATOM utils are built
  ExternalProject_Add_Step(LBANN build-atom-utils
    COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> --target atom-utils
    COMMENT "Performing building of ATOM tools for 'LBANN'"
    DEPENDEES build
    DEPENDERS install
    LOG 1
    USES_TERMINAL 1)

  # Ensure the PILOT2 utils are built
  ExternalProject_Add_Step(LBANN build-pilot2-utils
    COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> --target pilot2-utils
    COMMENT "Performing building of PILOT2 tools for 'LBANN'"
    DEPENDEES build
    DEPENDERS install
    LOG 1
    USES_TERMINAL 1)

endif ()

set(LBANN_DIR ${LBANN_SB_LBANN_PREFIX}
    CACHE INTERNAL "The install prefix of LBANN.")
