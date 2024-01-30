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
  Hydrogen
  Hydrogen_ENABLE_ALUMINUM
  "Enable Aluminum-based communication in Hydrogen"
  ${LBANN_SB_BUILD_Aluminum})

lbann_sb_default_pkg_option(Hydrogen
  Hydrogen_GENERAL_LAPACK_FALLBACK
  "Allow a fallback search for system blas/lapack libraries in Hydrogen"
  ON)

lbann_sb_default_pkg_option(
  Hydrogen
  Hydrogen_ENABLE_OPENMP
  "Use OpenMP threading (host code only) in Hydrogen"
  ON)

lbann_sb_default_pkg_option(
  Hydrogen
  Hydrogen_USE_64BIT_INTS
  "Use 64-bit indexing in Hydrogen"
  ON)

lbann_sb_default_cuda_option(
  Hydrogen
  Hydrogen_ENABLE_CUDA
  "Enable CUDA support in Hydrogen"
  ON)

lbann_sb_default_rocm_option(
  Hydrogen
  Hydrogen_ENABLE_ROCM
  "Enable ROCm support in Hydrogen"
  ON)

lbann_sb_default_gpu_option(
  Hydrogen
  Hydrogen_ENABLE_CUB
  "Enable (hip)CUB memory allocators in Hydrogen"
  ON)

lbann_sb_add_cmake_extern_pkg(
  NAME Hydrogen
  LANGUAGES CXX
  OPTIONAL_LANGUAGES CUDA HIP
  GITHUB_URL llnl/Elemental.git
  GIT_TAG "hydrogen"
  DEPENDS_ON Aluminum Catch2 OpenBLAS)

set(Hydrogen_DIR ${LBANN_SB_Hydrogen_PREFIX}
  CACHE INTERNAL "The install prefix of Hydrogen.")
