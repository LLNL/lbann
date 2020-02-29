#!/bin/sh

# Options
spack_dir=~/workspace/spack
lbann_dir=~/workspace/lbann
build_dir=$(realpath $(dirname $0))
install_dir=${build_dir}/install

# Move to build directory
mkdir -p ${build_dir}
mkdir -p ${install_dir}
pushd ${build_dir}

# Spack magic
. ${spack_dir}/share/spack/setup-env.sh
ml cmake/3.12.1 gcc/7.3.0 mvapich2/2.3 cuda/10.0.130
#spack env create -d . ${lbann_dir}/spack_environments/developer_release_x86_64_cuda_spack.yaml
spack env create -d . ${lbann_dir}/spack_environments/developer_release_cuda_spack.yaml
spack install
spack env loads
source loads
unset LIBRARY_PATH

echo "building LBANN"

# Build LBANN
cmake \
    -G Ninja \
    -D CMAKE_BUILD_TYPE:STRING=Release \
    -D CMAKE_INSTALL_PREFIX:PATH=${install_dir} \
    \
    -D LBANN_SB_BUILD_ALUMINUM=ON \
    -D ALUMINUM_ENABLE_MPI_CUDA=OFF \
    -D ALUMINUM_ENABLE_NCCL=ON \
    \
    -D LBANN_SB_BUILD_HYDROGEN=ON \
    -D Hydrogen_ENABLE_ALUMINUM=ON \
    -D Hydrogen_ENABLE_CUB=ON \
    -D Hydrogen_ENABLE_CUDA=ON \
    \
    -D LBANN_SB_BUILD_LBANN=ON \
    -D LBANN_DATATYPE:STRING=float \
    -D LBANN_SEQUENTIAL_INITIALIZATION:BOOL=OFF \
    -D LBANN_WITH_ALUMINUM:BOOL=ON \
    -D LBANN_WITH_CONDUIT:BOOL=OFF \
    -D LBANN_WITH_CUDA:BOOL=ON \
    -D LBANN_WITH_CUDNN:BOOL=ON \
    -D LBANN_WITH_NCCL:BOOL=ON \
    -D LBANN_WITH_NVPROF:BOOL=ON \
    -D LBANN_WITH_SOFTMAX_CUDA:BOOL=ON \
    -D LBANN_WITH_TOPO_AWARE:BOOL=ON \
    -D LBANN_WITH_TBINF=OFF \
    -D LBANN_WITH_VTUNE:BOOL=OFF \
    ${lbann_dir}/superbuild
ninja

# Return to original directory
dirs -c
