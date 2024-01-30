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
# Most of the folks using the superbuild are devs, and this has no
# performance implications for those who aren't.
lbann_sb_default_pkg_option(
  DiHydrogen
  H2_DEVELOPER_BUILD
  "Enable developer warnings/features in H2"
  ON)

# Explicitly acknowledge DistConv
lbann_sb_default_pkg_option(
  DiHydrogen
  H2_ENABLE_DISTCONV_LEGACY
  "Enable legacy DistConv library in DiHydrogen"
  OFF)

lbann_sb_default_cuda_option(
  DiHydrogen
  H2_ENABLE_CUDA
  "Enable CUDA support in DiHydrogen"
  ON)

lbann_sb_default_rocm_option(
  DiHydrogen
  H2_ENABLE_ROCM
  "Enable HIP/ROCm support in DiHydrogen"
  ON)

lbann_sb_add_cmake_extern_pkg(
  NAME DiHydrogen
  LANGUAGES CXX
  OPTIONAL_LANGUAGES CUDA HIP
  GITHUB_URL LLNL/DiHydrogen.git
  GIT_TAG "develop"
  DEPENDS_ON Aluminum Catch2 spdlog Hydrogen
)

set(DiHydrogen_DIR ${LBANN_SB_DiHydrogen_PREFIX}
  CACHE INTERNAL "The install prefix of DiHydrogen.")
