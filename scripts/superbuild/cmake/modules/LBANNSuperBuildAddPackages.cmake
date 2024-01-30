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
include(CMakeDependentOption)

macro(lbann_sb_default_pkg_option PKG_NAME OPTION_NAME DOC_STR VALUE)
  option(LBANN_SB_FWD_${PKG_NAME}_${OPTION_NAME}
    "${DOC_STR}"
    ${VALUE})
endmacro ()

macro(lbann_sb_default_cuda_option PKG_NAME OPTION_NAME DOC_STR VALUE)
  cmake_dependent_option(
    LBANN_SB_FWD_${PKG_NAME}_${OPTION_NAME}
    "${DOC_STR}"
    ${VALUE}
    "LBANN_SB_DEFAULT_CUDA_OPTS"
    OFF)
endmacro ()

macro(lbann_sb_default_rocm_option PKG_NAME OPTION_NAME DOC_STR VALUE)
  cmake_dependent_option(
    LBANN_SB_FWD_${PKG_NAME}_${OPTION_NAME}
    "${DOC_STR}"
    ${VALUE}
    "LBANN_SB_DEFAULT_ROCM_OPTS"
    OFF)
endmacro ()

macro(lbann_sb_default_gpu_option PKG_NAME OPTION_NAME DOC_STR VALUE)
  cmake_dependent_option(
    LBANN_SB_FWD_${PKG_NAME}_${OPTION_NAME}
    "${DOC_STR}"
    ${VALUE}
    "LBANN_SB_DEFAULT_CUDA_OPTS OR LBANN_SB_DEFAULT_ROCM_OPTS"
    OFF)
endmacro ()

macro(lbann_sb_add_package PKG_NAME)
  option(LBANN_SB_BUILD_${PKG_NAME}
    "Optionally download and build ${PKG_NAME}?"
    OFF)
  if (LBANN_SB_BUILD_${PKG_NAME})
    list(APPEND LBANN_SB_BUILD_PKGS ${PKG_NAME})
  endif ()
endmacro ()

macro(lbann_sb_add_packages)
  foreach (pkg ${ARGN})
    lbann_sb_add_package(${pkg})
  endforeach ()
endmacro ()
