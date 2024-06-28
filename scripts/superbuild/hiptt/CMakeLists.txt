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
lbann_sb_init_extern_pkg(
  NAME hiptt
  LANGUAGES C CXX HIP
  GITHUB_URL tbennun/hipTT.git
  GIT_TAG "master")

# The build system here is just a set of makefiles.
find_program(GNU_MAKE_PROGRAM make)

include (ExternalProject)
ExternalProject_Add(${PKG_NAME}
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
  ${LBANN_SB_GIT_REPOSITORY_TAG} ${LBANN_SB_${PKG_NAME}_URL}
  ${LBANN_SB_GIT_TAG_TAG} ${LBANN_SB_${PKG_NAME}_TAG}
  TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp"
  STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/stamp"

  SOURCE_DIR "${LBANN_SB_${PKG_NAME}_SOURCE_DIR}"
  INSTALL_DIR "${LBANN_SB_${PKG_NAME}_PREFIX}"

  GIT_SHALLOW 1

  BUILD_IN_SOURCE 1
  USES_TERMINAL_BUILD 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_BUILD 1
  LOG_INSTALL 1
  LOG_TEST 1

  CONFIGURE_COMMAND ""

  BUILD_COMMAND
  ${GNU_MAKE_PROGRAM}
  "PREFIX=${LBANN_SB_${PKG_NAME}_PREFIX}"
  "CC=${LBANN_SB_${PKG_NAME}_C_COMPILER}"
  "CXX=${LBANN_SB_${PKG_NAME}_CXX_COMPILER}"
  -j${${PKG_NAME}_MAX_MAKE_JOBS}

  INSTALL_COMMAND
  ${CMAKE_COMMAND} -E copy_directory
  ${LBANN_SB_${PKG_NAME}_SOURCE_DIR}/include
  ${LBANN_SB_${PKG_NAME}_PREFIX}/include

  COMMAND
  ${CMAKE_COMMAND} -E copy_directory
  ${LBANN_SB_${PKG_NAME}_SOURCE_DIR}/lib
  ${LBANN_SB_${PKG_NAME}_PREFIX}/lib

)

set(${PKG_NAME}_DIR ${LBANN_SB_${PKG_NAME}_PREFIX}
  CACHE INTERNAL "The install prefix of ${PKG_NAME}.")
