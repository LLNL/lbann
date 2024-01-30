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
# Caliper-specific configuration options explicitly exposed
lbann_sb_default_pkg_option(
  Caliper
  WITH_ADIAK
  "Enable adiak support in Caliper."
  ${LBANN_SB_BUILD_adiak})

lbann_sb_default_pkg_option(
  Caliper
  WITH_FORTRAN
  "Enable fortran support in Caliper."
  OFF)

lbann_sb_default_pkg_option(
  Caliper
  WITH_OMPT
  "Enable OMPT support in Caliper."
  OFF)

lbann_sb_default_pkg_option(
  Caliper
  WITH_MPI
  "Enable MPI support in Caliper."
  ON)

lbann_sb_default_cuda_option(
  Caliper
  WITH_CUPTI
  "Enable cupty support in Caliper."
  ON)

lbann_sb_default_cuda_option(
  Caliper
  WITH_NVTX
  "Enable nvtx support in Caliper."
  ON)

lbann_sb_default_rocm_option(
  Caliper
  WITH_ROCM
  "Enable ROCm support in Caliper."
  ON)

lbann_sb_default_rocm_option(
  Caliper
  WITH_ROCTRACER
  "Enable roctracer support in Caliper."
  ON)

lbann_sb_default_rocm_option(
  Caliper
  WITH_ROCTX
  "Enable roctx support in Caliper."
  ON)

set(lbann_sb_ftn_lang)
if (LBANN_SB_FWD_Caliper_WITH_FORTRAN)
  set(lbann_sb_ftn_lang Fortran)
endif ()

lbann_sb_add_cmake_extern_pkg(
  NAME Caliper
  LANGUAGES C CXX ${lbann_sb_ftn_lang}
  GITHUB_URL llnl/caliper.git
  GIT_TAG "master"
  DEPENDS_ON adiak
)

set(Caliper_DIR ${LBANN_SB_Caliper_PREFIX}
  CACHE INTERNAL "The install prefix of Caliper.")
