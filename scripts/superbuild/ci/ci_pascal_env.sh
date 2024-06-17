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

# Improve debugging info and remove some misguided warnings. These are
# passed only to the LBANN stack.
EXTRA_CXX_FLAGS="-g3 -Wno-deprecated-declarations"
EXTRA_CUDA_FLAGS="-g3 -Wno-deprecated-declarations"

# Prefer RPATH to RUNPATH (stability over flexibility)
EXTRA_LINK_FLAGS="-Wl,--disable-new-dtags"

# Set this to the CUDA GPU arch(s) to support (example set for Lassen/Sierra)
CUDA_GPU_ARCH=60

# Set to the preferred install directory
INSTALL_ROOT=/usr/workspace/lbann/ci_stable_dependencies/pascal/cuda-11.8.0
INSTALL_PREFIX_EXTERNALS=${INSTALL_ROOT}/openmpi-4.1.2

# Location of external packages
export CMAKE_PREFIX_PATH=${INSTALL_ROOT}/cudnn-8.9.4:${INSTALL_ROOT}/nccl-2.19.4
CMAKE_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH//:/;}
