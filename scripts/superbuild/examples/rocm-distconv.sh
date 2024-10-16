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

# Set to ON (or any CMake truthy value) to build all of the
# dependencies of the LBANN stack
BUILD_EXTERNAL_TPLS=ON

# Set to ON to build Aluminum, Hydrogen, DiHydrogen, and LBANN
BUILD_LBANN_STACK=ON

# Set to ON to enable DistConv support. Only matters if building the
# LBANN stack.
BUILD_WITH_DISTCONV=ON

# Set to ON if you're on a Cray machine that doesn't provide the AWS
# plugin as part of its default RCCL installation.
#
# It might also be advisable to build this if you build a custom RCCL.
# The configuration script takes a RCCL path as a parameter, so it
# could matter, but it's not clear how much.
BUILD_AWS_OFI_RCCL_PLUGIN=OFF

# Improve debugging info and remove some misguided warnings. These are
# passed only to the LBANN stack.
EXTRA_CXX_FLAGS="-g3 -Wno-deprecated-declarations"
EXTRA_HIP_FLAGS="-g3 -Wno-deprecated-declarations"

# Prefer RPATH to RUNPATH (stability over flexibility)
EXTRA_LINK_FLAGS="-Wl,--disable-new-dtags"

# Set this to the AMD GPU arch(s) to support (example set for Crusher/Frontier/Tioga)
AMD_GPU_ARCH=gfx90a

# Set to the directory with the top-level CMakeLists.txt file for LBANN
LBANN_SRC_DIR=$(git rev-parse --show-toplevel)

# Set to the directory with the top-level SuperBuild CMakeLists.txt file
SUPERBUILD_SRC_DIR=${LBANN_SRC_DIR}/scripts/superbuild

# Set to the preferred install directory
INSTALL_PREFIX=${PWD}/install-rocm-distconv

# Set to the preferred build directory
BUILD_DIR=${TMPDIR}/lbann-superbuild

cmake \
    -G Ninja \
    -S ${SUPERBUILD_SRC_DIR} \
    -B ${BUILD_DIR} \
    \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    \
    -D CMAKE_C_COMPILER=$(which amdclang) \
    -D CMAKE_CXX_COMPILER=$(which amdclang++) \
    -D CMAKE_Fortran_COMPILER=$(which gfortran) \
    \
    -D CMAKE_EXE_LINKER_FLAGS=${EXTRA_LINK_FLAGS} \
    -D CMAKE_SHARED_LINKER_FLAGS=${EXTRA_LINK_FLAGS} \
    \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_HIP_STANDARD=17 \
    -D CMAKE_HIP_ARCHITECTURES=${AMD_GPU_ARCH} \
    \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
    \
    -D LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY="PKG_LC" \
    -D LBANN_SB_DEFAULT_ROCM_OPTS=ON \
    \
    -D LBANN_SB_BUILD_adiak=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_Caliper=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_adiak_BUILD_SHARED_LIBS=ON \
    -D LBANN_SB_Caliper_BUILD_SHARED_LIBS=ON \
    \
    -D LBANN_SB_BUILD_Catch2=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_cereal=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_Clara=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_CNPY=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_protobuf=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_spdlog=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_zstr=${BUILD_EXTERNAL_TPLS} \
    \
    -D LBANN_SB_BUILD_Conduit=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_HDF5=${BUILD_EXTERNAL_TPLS} \
    \
    -D LBANN_SB_BUILD_JPEG-TURBO=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_OpenCV=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_OpenCV_TAG=4.x \
    \
    -D LBANN_SB_BUILD_AWS_OFI_RCCL=${BUILD_AWS_OFI_RCCL_PLUGIN} \
    \
    -D LBANN_SB_BUILD_Aluminum=${BUILD_LBANN_STACK} \
    -D LBANN_SB_Aluminum_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -D LBANN_SB_Aluminum_HIP_FLAGS="${EXTRA_HIP_FLAGS}" \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_CALIPER=ON \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_NCCL=ON \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_HOST_TRANSFER=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_TESTS=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_BENCHMARKS=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_THREAD_MULTIPLE=OFF \
    \
    -D LBANN_SB_BUILD_Hydrogen=${BUILD_LBANN_STACK} \
    -D LBANN_SB_Hydrogen_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -D LBANN_SB_Hydrogen_HIP_FLAGS="${EXTRA_HIP_FLAGS}" \
    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_HALF=OFF \
    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_TESTING=ON \
    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_UNIT_TESTS=OFF \
    \
    -D LBANN_SB_BUILD_DiHydrogen=${BUILD_LBANN_STACK} \
    -D LBANN_SB_DiHydrogen_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -D LBANN_SB_DiHydrogen_HIP_FLAGS="${EXTRA_HIP_FLAGS}" \
    -D LBANN_SB_FWD_DiHydrogen_H2_ENABLE_DISTCONV_LEGACY=${BUILD_WITH_DISTCONV} \
    \
    -D LBANN_SB_BUILD_LBANN=${BUILD_LBANN_STACK} \
    -D LBANN_SB_LBANN_BUILD_SHARED_LIBS=ON \
    -D LBANN_SB_LBANN_SOURCE_DIR=${LBANN_SRC_DIR} \
    -D LBANN_SB_FWD_LBANN_CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -D LBANN_SB_LBANN_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -D LBANN_SB_LBANN_HIP_FLAGS="${EXTRA_HIP_FLAGS}" \
    -D LBANN_SB_FWD_LBANN_LBANN_DATATYPE=float \
    -D LBANN_SB_FWD_LBANN_LBANN_WITH_CALIPER=ON \
    -D LBANN_SB_FWD_LBANN_LBANN_WITH_DISTCONV=${BUILD_WITH_DISTCONV} \
    -D LBANN_SB_FWD_LBANN_LBANN_WITH_TBINF=OFF \
    -D LBANN_SB_FWD_LBANN_LBANN_WITH_UNIT_TESTING=ON
