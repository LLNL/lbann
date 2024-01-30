################################################################################
## Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
  NAME zstr
  LANGUAGES NONE
  GITHUB_URL mateidavid/zstr
  GIT_TAG master)

# Now add the external project
include(ExternalProject)
ExternalProject_Add(zstr
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
  STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build

  SOURCE_DIR ${LBANN_SB_zstr_SOURCE_DIR}
  INSTALL_DIR ${LBANN_SB_zstr_PREFIX}

  ${LBANN_SB_GIT_REPOSITORY_TAG} ${LBANN_SB_zstr_URL}
  ${LBANN_SB_GIT_TAG_TAG} ${LBANN_SB_zstr_TAG}

  USES_TERMINAL_INSTALL 1
  LOG_DOWNLOAD 1
  LOG_UPDATE 1
  LOG_INSTALL 1

  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""

  INSTALL_COMMAND
  ${CMAKE_COMMAND} -E copy_directory
  ${LBANN_SB_zstr_SOURCE_DIR}/src
  ${LBANN_SB_zstr_PREFIX}/include
)

set(zstr_DIR ${LBANN_SB_zstr_PREFIX}
  CACHE INTERNAL "The install prefix of zstr.")
