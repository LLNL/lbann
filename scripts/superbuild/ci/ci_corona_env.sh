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
EXTRA_HIP_FLAGS="-g3 -Wno-deprecated-declarations"

# Prefer RPATH to RUNPATH (stability over flexibility)
EXTRA_LINK_FLAGS="-fuse-ld=lld -Wl,--disable-new-dtags"
# If using PrgEnv-cray add ${CRAYLIBS_X86_64}
EXTRA_RPATHS="${ROCM_PATH}/lib|${ROCM_PATH}/llvm/lib"

# Set this to the AMD GPU arch(s) to support (example set for Crusher/Frontier/Tioga)
AMD_GPU_ARCH=gfx906

ROCM_VER=$(basename ${ROCM_PATH})
COMPILER_VER=clang-14.0.6-magic
# Set to the preferred install directory
CI_STABLE_DEPENDENCIES_ROOT=/usr/workspace/lbann/ci_stable_dependencies
INSTALL_ROOT=${CI_STABLE_DEPENDENCIES_ROOT}/corona/${ROCM_VER}
INSTALL_PREFIX_EXTERNALS=${INSTALL_ROOT}/${COMPILER_VER}/openmpi-4.1.2

# Use an accessible build directory so that the source files are preserved for debugging
BUILD_ROOT=/usr/workspace/lbann/ci_stable_dependencies/.build/corona/${ROCM_VER}/${COMPILER_VER}

# Location of external packages
CMAKE_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH//:/;}

#export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
