#!/bin/sh

LBANN_FWD_CMD=
LBANN_COMPILER_CMD=

if [[ ${SYS} = "Darwin" ]]; then
LBANN_FWD_CMD=$(cat << EOF
  -D HWLOC_DIR=/usr/local/opt/hwloc \
  -D OpenMP_CXX_LIB_NAMES=omp \
  -D OpenMP_CXX_FLAGS="-fopenmp=libomp" \
  -D OpenMP_omp_LIBRARY=/usr/local/opt/llvm/lib/libomp.dylib
EOF
)
LBANN_COMPILER_CMD=$(cat << EOF
  -D CMAKE_CXX_COMPILER=/opt/rocm-3.8.0/hip/bin/hipcc \
  -D CMAKE_C_COMPILER=$(which clang)
EOF
)
fi

# Configure build with CMake
CONFIGURE_COMMAND=$(cat << EOF
cmake \
  -G Ninja \
  -D CMAKE_CXX_COMPILER=/opt/rocm-3.8.0/hip/bin/hipcc \
  -D CMAKE_MPI_CXX_COMPILER=/opt/rocm-3.8.0/hip/bin/hipcc \
  -D CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE} \
  -D CMAKE_INSTALL_PREFIX:PATH=${LBANN_INSTALL_DIR} \
  -D CMAKE_CXX_FLAGS="${CXX_FLAGS} -g -fPIC -shared -fsized-deallocation -std=c++17 -fno-gpu-rdc -Wno-deprecated-declarations -Wno-unused-command-line-argument" \
  -D CMAKE_C_FLAGS="${C_FLAGS} -fPIC -fsized-deallocation" \
  -D HIP_HIPCC_FLAGS="-fPIC -shared -fsized-deallocation -std=c++17" \
  \
  -D LBANN_DATATYPE:STRING=float \
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
  -D LBANN_WITH_DIHYDROGEN=${ENABLE_DIHYDROGEN} \
  -D LBANN_WITH_DISTCONV:BOOL=${ENABLE_DISTCONV} \
  \
  -D LBANN_WITH_EMBEDDED_PYTHON=${ENABLE_EMBEDDED_PYTHON} \
  -D LBANN_WITH_PYTHON_FRONTEND=${ENABLE_PYTHON_FRONTEND} \
${LBANN_FWD_CMD}  \
${LBANN_COMPILER_CMD}  \
  ${LBANN_HOME}
EOF
)

if [[ ${VERBOSE} -ne 0 ]]; then
    echo "${CONFIGURE_COMMAND}" 2>1 | tee lbann_cmake_invocation.txt
else
    echo "${CONFIGURE_COMMAND}" > lbann_cmake_invocation.txt
fi
eval ${CONFIGURE_COMMAND}
if [[ $? -ne 0 ]]; then
    echo "--------------------"
    echo "CONFIGURE FAILED"
    echo "--------------------"
    exit 1
fi
