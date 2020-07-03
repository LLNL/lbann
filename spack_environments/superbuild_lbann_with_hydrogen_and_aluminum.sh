#!/bin/sh

AL_FWD_CMD=
HYDROGEN_FWD_CMD=
LBANN_FWD_CMD=
LBANN_COMPILER_CMD=

if [[ ${SYS} = "Darwin" ]]; then
AL_FWD_CMD=$(cat << EOF
  -D LBANN_SB_FWD_ALUMINUM_OpenMP_CXX_LIB_NAMES=omp \
  -D LBANN_SB_FWD_ALUMINUM_OpenMP_CXX_FLAGS=-fopenmp \
  -D LBANN_SB_FWD_ALUMINUM_OpenMP_omp_LIBRARY=/usr/local/opt/llvm/lib/libomp.dylib
EOF
)
HYDROGEN_FWD_CMD=$(cat << EOF
  -D LBANN_SB_FWD_HYDROGEN_OpenMP_CXX_LIB_NAMES=omp \
  -D LBANN_SB_FWD_HYDROGEN_OpenMP_CXX_FLAGS="-fopenmp=libomp" \
  -D LBANN_SB_FWD_HYDROGEN_OpenMP_omp_LIBRARY=/usr/local/opt/llvm/lib/libomp.dylib
EOF
)
LBANN_FWD_CMD=$(cat << EOF
  -D LBANN_SB_FWD_LBANN_HWLOC_DIR=/usr/local/opt/hwloc \
  -D LBANN_SB_FWD_LBANN_OpenMP_CXX_LIB_NAMES=omp \
  -D LBANN_SB_FWD_LBANN_OpenMP_CXX_FLAGS="-fopenmp=libomp" \
  -D LBANN_SB_FWD_LBANN_OpenMP_omp_LIBRARY=/usr/local/opt/llvm/lib/libomp.dylib
EOF
)
LBANN_COMPILER_CMD=$(cat << EOF
  -D CMAKE_CXX_COMPILER=$(which clang++) \
  -D CMAKE_C_COMPILER=$(which clang)
EOF
)

fi

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
  -D ALUMINUM_TAG=v0.3.3 \
  -D ALUMINUM_ENABLE_MPI_CUDA=OFF \
  -D ALUMINUM_ENABLE_NCCL=${ENABLE_GPUS} \
${AL_FWD_CMD}  \
  \
  -D LBANN_SB_BUILD_HYDROGEN=ON \
  -D Hydrogen_ENABLE_ALUMINUM=ON \
  -D Hydrogen_ENABLE_CUB=${ENABLE_GPUS} \
  -D Hydrogen_ENABLE_CUDA=${ENABLE_GPUS} \
  -D Hydrogen_ENABLE_HALF=ON \
${HYDROGEN_FWD_CMD}  \
  \
  -D LBANN_SB_BUILD_LBANN=ON \
  -D LBANN_DATATYPE:STRING=float \
  -D LBANN_SEQUENTIAL_INITIALIZATION:BOOL=OFF \
  -D LBANN_WITH_ALUMINUM:BOOL=ON \
  -D LBANN_WITH_CONDUIT:BOOL=ON \
  -D LBANN_WITH_CUDA:BOOL=${ENABLE_GPUS} \
  -D LBANN_WITH_CUDNN:BOOL=${ENABLE_GPUS} \
  -D LBANN_WITH_NCCL:BOOL=${ENABLE_GPUS} \
  -D LBANN_WITH_NVPROF:BOOL=${ENABLE_GPUS} \
  -D LBANN_WITH_SOFTMAX_CUDA:BOOL=${ENABLE_GPUS} \
  -D LBANN_WITH_TOPO_AWARE:BOOL=ON \
  -D LBANN_WITH_TBINF=OFF \
  -D LBANN_WITH_VTUNE:BOOL=OFF \
  -D LBANN_DETERMINISTIC=${DETERMINISTIC} \
${LBANN_FWD_CMD}  \
${LBANN_COMPILER_CMD}  \
  ${LBANN_HOME}/superbuild
EOF
)

if [[ ${VERBOSE} -ne 0 ]]; then
    echo "${CONFIGURE_COMMAND}" 2>1 | tee cmake_superbuild_invocation.txt
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
