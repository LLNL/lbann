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
EXTRA_CXX_FLAGS="-O2 -g3 -Wno-deprecated-declarations"
EXTRA_HIP_FLAGS="-O2 -g3 -Wno-deprecated-declarations"

# Prefer RPATH to RUNPATH (stability over flexibility)
EXTRA_LINK_FLAGS="-fuse-ld=lld -Wl,--disable-new-dtags"
# If using PrgEnv-cray add ${CRAYLIBS_X86_64}
EXTRA_RPATHS="${ROCM_PATH}/lib\;${ROCM_PATH}/llvm/lib"

# Set this to the AMD GPU arch(s) to support (example set for Crusher/Frontier/Tioga)
AMD_GPU_ARCH=gfx90a,gfx942

# Set to the preferred install directory
#INSTALL_PREFIX=${PWD}/install-rocm-distconv
ROCM_VER=$(basename ${ROCM_PATH})
PE_ENV_lc=$(echo "${PE_ENV}" | tr '[:upper:]' '[:lower:]')
INSTALL_ROOT=/usr/workspace/lbann/ci_stable_dependencies/tioga/${ROCM_VER}/${PE_ENV_lc}
INSTALL_PREFIX_EXTERNALS=${INSTALL_ROOT}/cray-mpich-8.1.29/

# Use an accessible build directory so that the source files are preserved for debuggin
BUILD_ROOT=/usr/workspace/lbann/ci_stable_dependencies/.build/tioga/${ROCM_VER}/${PE_ENV_lc}

# Location of external packages
#export CMAKE_PREFIX_PATH=/p/vast1/lbann/stable_dependencies/tioga/rocm-6.1.0/miopen
CMAKE_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH//:/;}
# Patch up the header while dealing with MiOpen
#export CPLUS_INCLUDE_PATH=/p/vast1/lbann/stable_dependencies/tioga/rocm-6.1.0/miopen/include:${CPLUS_INCLUDE_PATH}

#CPLUS_INCLUDE_PATH=/p/vast1/lbann/stable_dependencies/tioga/rocm-6.1.0/miopen/include:${CPLUS_INCLUDE_PATH}

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
