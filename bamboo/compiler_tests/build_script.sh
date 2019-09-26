#!/bin/bash

source /usr/share/lmod/lmod/init/bash
source /etc/profile.d/00-modulepath.sh

LBANN_DIR=$(git rev-parse --show-toplevel)
CLUSTER=$(hostname | sed 's/[0-9]*//g')
USER=$(whoami)
WORKSPACE_DIR=$(ls --color=no -d /usr/workspace/*/${USER})
DEPENDENCY_DIR=${WORKSPACE_DIR}/stable_dependencies/${CLUSTER}/gcc-7.3.0/mvapich2-2.3
if [ -e ${DEPENDENCY_DIR} ];
then
    ml gcc/7.3.0 cmake/3.14.5 cuda
    export NCCL_DIR=/usr/workspace/wsb/brain/nccl2/nccl_2.4.2-1+cuda10.0_x86_64
    export CUDNN_DIR=/usr/WS2/brain/cudnn/cudnn-7.6.2/cuda-10.1_x86_64
    export PATH=${WORKSPACE_DIR}/stable_dependencies/${CLUSTER}/ninja/bin:${PATH}

    # Setup paths to match the build script (ugh)
    BUILD_DIR_BASE=${LBANN_DIR}/build/gnu.Release.${CLUSTER}.llnl.gov
    BUILD_DIR=${BUILD_DIR_BASE}/lbann/build
    INSTALL_DIR=${BUILD_DIR_BASE}/install
    
    # Setup a path for Catch2 to use
    CATCH2_OUTPUT_DIR=${LBANN_DIR}/bamboo/compiler_tests
    
    # Cleanup
    [ -e ${BUILD_DIR_BASE} ] && rm -rf ${BUILD_DIR_BASE}
    mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
    
    cmake \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
        \
        -DCMAKE_CXX_COMPILER=$(which g++) \
        -DCMAKE_CXX_FLAGS="-DLBANN_SET_EL_RNG -g" \
        -DCMAKE_CUDA_COMPILER=$(which nvcc) \
        -DCMAKE_CUDA_HOST_COMPILER=$(which g++) \
        \
        -DCMAKE_CXX_STANDARD=14 \
        -DCMAKE_CUDA_STANDARD=14 \
        \
        -DLBANN_DATATYPE=float \
        -DLBANN_DETERMINISTIC=OFF \
        -DLBANN_WARNINGS_AS_ERRORS=ON \
        -DLBANN_WITH_ALUMINUM=ON \
        -DLBANN_WITH_CONDUIT=ON \
        -DLBANN_WITH_CUDA=ON \
        -DLBANN_WITH_NVPROF=OFF \
        -DLBANN_WITH_TBINF=ON \
        -DLBANN_WITH_UNIT_TESTING=ON \
        -DLBANN_WITH_VTUNE=OFF \
        \
        -DAluminum_DIR=${DEPENDENCY_DIR}/lib/cmake/Aluminum \
        -DCEREAL_DIR=${DEPENDENCY_DIR} \
        -DCNPY_DIR=${DEPENDENCY_DIR} \
        -DCATCH2_DIR=${WORKSPACE_DIR}/stable_dependencies/catch2 \
        -DHDF5_DIR=${DEPENDENCY_DIR} \
        -DCONDUIT_DIR=${DEPENDENCY_DIR} \
        -DCUB_DIR=${DEPENDENCY_DIR} \
        -DHydrogen_DIR=${DEPENDENCY_DIR} \
        -DOpenCV_DIR=${DEPENDENCY_DIR} \
        -DPROTOBUF_DIR=${DEPENDENCY_DIR} \
        -Dprotobuf_MODULE_COMPATIBLE=ON \
        \
        ${LBANN_DIR} && ninja && ninja install && ./unit_test/seq-catch-tests -r junit -o ${CATCH2_OUTPUT_DIR}/seq_catch_tests_output.xml
else
    ${LBANN_DIR}/scripts/build_lbann_lc.sh --with-conduit
fi
