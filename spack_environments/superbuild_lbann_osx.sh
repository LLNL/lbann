#!/bin/bash

# Configure build with CMake
CONFIGURE_COMMAND=$(cat << EOF
cmake \
  -G Ninja \
  -D CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE} \
  -D CMAKE_INSTALL_PREFIX:PATH=${LBANN_INSTALL_DIR} \
  -D CMAKE_CXX_FLAGS="${CXX_FLAGS}" \
  -D CMAKE_C_FLAGS="${C_FLAGS}" \
  \
  -D LBANN_SB_BUILD_ALUMINUM=ON \
  -D ALUMINUM_ENABLE_MPI_CUDA=OFF \
  -D ALUMINUM_ENABLE_NCCL=OFF \
  -D LBANN_SB_FWD_ALUMINUM_OpenMP_CXX_LIB_NAMES=omp \
  -D LBANN_SB_FWD_ALUMINUM_OpenMP_CXX_FLAGS=-fopenmp \
  -D LBANN_SB_FWD_ALUMINUM_OpenMP_omp_LIBRARY=/usr/local/opt/llvm/lib/libomp.dylib \
  \
  -D LBANN_SB_BUILD_HYDROGEN=ON \
  -D Hydrogen_ENABLE_ALUMINUM=ON \
  -D Hydrogen_ENABLE_CUB=OFF \
  -D Hydrogen_ENABLE_CUDA=OFF \
  -D Hydrogen_ENABLE_HALF=ON \
  -D LBANN_SB_FWD_HYDROGEN_OpenMP_CXX_LIB_NAMES=omp \
  -D LBANN_SB_FWD_HYDROGEN_OpenMP_CXX_FLAGS="-fopenmp=libomp" \
  -D LBANN_SB_FWD_HYDROGEN_OpenMP_omp_LIBRARY=/usr/local/opt/llvm/lib/libomp.dylib \
  \
  -D LBANN_SB_BUILD_LBANN=ON \
  -D LBANN_DATATYPE:STRING=float \
  -D LBANN_SEQUENTIAL_INITIALIZATION:BOOL=OFF \
  -D LBANN_WITH_ALUMINUM:BOOL=ON \
  -D LBANN_WITH_CONDUIT:BOOL=ON \
  -D LBANN_WITH_CUDA:BOOL=OFF \
  -D LBANN_WITH_CUDNN:BOOL=OFF \
  -D LBANN_WITH_NCCL:BOOL=OFF \
  -D LBANN_WITH_NVPROF:BOOL=OFF \
  -D LBANN_WITH_SOFTMAX_CUDA:BOOL=OFF \
  -D LBANN_WITH_TOPO_AWARE:BOOL=ON \
  -D LBANN_WITH_TBINF=OFF \
  -D LBANN_WITH_VTUNE:BOOL=OFF \
  -D LBANN_DETERMINISTIC=${DETERMINISTIC} \
  -D LBANN_SB_FWD_LBANN_HWLOC_DIR=/usr/local/opt/hwloc \
  -D LBANN_SB_FWD_LBANN_OpenMP_CXX_LIB_NAMES=omp \
  -D LBANN_SB_FWD_LBANN_OpenMP_CXX_FLAGS="-fopenmp=libomp" \
  -D LBANN_SB_FWD_LBANN_OpenMP_omp_LIBRARY=/usr/local/opt/llvm/lib/libomp.dylib \
  \
  -D CMAKE_CXX_COMPILER=$(which clang++) \
  -D CMAKE_C_COMPILER=$(which clang) \
  ${LBANN_HOME}/superbuild
EOF
)

#echo ${CONFIGURE_COMMAND}

if [[ ${VERBOSE} -ne 0 ]]; then
    echo "${CONFIGURE_COMMAND}" 2>&1 | tee cmake_superbuild_invocation.txt
else
    echo "${CONFIGURE_COMMAND}" > cmake_superbuild_invocation.txt
fi
eval ${CONFIGURE_COMMAND}
if [[ $? -ne 0 ]]; then
    echo "--------------------"
    echo "CONFIGURE FAILED"
    echo "--------------------"
    exit 1
fi
