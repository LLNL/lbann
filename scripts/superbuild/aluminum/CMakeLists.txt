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
lbann_sb_default_pkg_option(
  Aluminum
  ALUMINUM_ENABLE_CALIPER
  "Enable Caliper support in Aluminum"
  ${LBANN_SB_BUILD_Caliper})

lbann_sb_default_cuda_option(
  Aluminum
  ALUMINUM_ENABLE_CUDA
  "Enable the base CUDA support in Aluminum"
  ON)

lbann_sb_default_rocm_option(
  Aluminum
  ALUMINUM_ENABLE_ROCM
  "Enable the base ROCM support in Aluminum"
  ON)

lbann_sb_default_gpu_option(
  Aluminum
  ALUMINUM_ENABLE_HOST_TRANSFER
  "Enable the host-transfer backend in Aluminum"
  ON)

if (LBANN_SB_BUILD_RCCL)
  lbann_sb_default_rocm_option(
    Aluminum
    ALUMINUM_ENABLE_NCCL
    "Enable RCCL support in Aluminum"
    ON)
endif ()

lbann_sb_add_cmake_extern_pkg(
  NAME Aluminum
  LANGUAGES CXX
  OPTIONAL_LANGUAGES CUDA HIP
  GITHUB_URL llnl/Aluminum.git
  GIT_TAG "master"
  DEPENDS_ON Caliper NCCL RCCL)

set(Aluminum_DIR ${LBANN_SB_Aluminum_PREFIX}
  CACHE INTERNAL "The install prefix of Aluminum.")
