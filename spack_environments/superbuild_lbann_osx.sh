#! /bin/bash

CMD="cmake \
       -G Ninja \
       -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
       -D CMAKE_BUILD_TYPE:STRING=Release \
       -D CMAKE_INSTALL_PREFIX:PATH=${LBANN_INSTALL_DIR} \
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
       -D LBANN_SB_FWD_LBANN_HWLOC_DIR=/usr/local/opt/hwloc \
       -D LBANN_SB_FWD_LBANN_OpenMP_CXX_LIB_NAMES=omp \
       -D LBANN_SB_FWD_LBANN_OpenMP_CXX_FLAGS="-fopenmp=libomp" \
       -D LBANN_SB_FWD_LBANN_OpenMP_omp_LIBRARY=/usr/local/opt/llvm/lib/libomp.dylib \
       \
       -D CMAKE_CXX_COMPILER=$(which clang++) \
       -D CMAKE_C_COMPILER=$(which clang) \
       ${LBANN_HOME}/superbuild"

echo ${CMD}
${CMD}
