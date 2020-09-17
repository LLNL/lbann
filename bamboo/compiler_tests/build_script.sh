#!/bin/bash

source /usr/share/lmod/lmod/init/bash
source /etc/profile.d/00-modulepath.sh

LBANN_DIR=$(git rev-parse --show-toplevel)
CLUSTER=$(hostname | sed 's/[0-9]*//g')
USER=$(whoami)
WORKSPACE_DIR=$(ls --color=no -d /usr/workspace/ws*/${USER})
COMMON_DEPENDENCY_DIR=${WORKSPACE_DIR}/stable_dependencies
DEPENDENCY_DIR_BASE=${COMMON_DEPENDENCY_DIR}/${CLUSTER}

# For this script, we only care about GCC.
LATEST_GCC=$(ls -1 ${DEPENDENCY_DIR_BASE} | grep gcc | tail -n1)
COMPILER_DIR=${DEPENDENCY_DIR_BASE}/${LATEST_GCC}

# For now, there's only one MPI library. The pipe to tail ensures that
# we just pick one thing, just in case.
MPI_LIBRARY=$(ls -1 --color=no ${COMPILER_DIR} | tail -n1)
MPI_DIR=${COMPILER_DIR}/${MPI_LIBRARY}

# All the dependencies are installed at the MPI level (even though
# most are MPI-independent).
DEPENDENCY_DIR=${MPI_DIR}

export CMAKE_PREFIX_PATH=${COMMON_DEPENDENCY_DIR}/catch2:${COMMON_DEPENDENCY_DIR}/cereal:${COMMON_DEPENDENCY_DIR}/clara:${COMMON_DEPENDENCY_DIR}/cub:${COMMON_DEPENDENCY_DIR}/half:${DEPENDENCY_DIR}/aluminum-0.4.0:${DEPENDENCY_DIR}/cnpy:${DEPENDENCY_DIR}/conduit:${DEPENDENCY_DIR}/hdf5:${DEPENDENCY_DIR}/hydrogen-1.4.0:${DEPENDENCY_DIR}/jpeg-turbo:${DEPENDENCY_DIR}/nccl:${DEPENDENCY_DIR}/openblas:${DEPENDENCY_DIR}/opencv:${DEPENDENCY_DIR}/protobuf:${CMAKE_PREFIX_PATH}

if [ -e ${DEPENDENCY_DIR} ];
then
    SAVELIST_NAME=$(echo ${CLUSTER}_${LATEST_GCC}_${MPI_LIBRARY} | sed -e 's/\./x/g')

    if ml -t savelist |& grep ${SAVELIST_NAME} > /dev/null 2>&1
    then
        ml restore ${SAVELIST_NAME}
    else
        # Compilers are easy...
        COMPILER_MODULE=$(echo ${LATEST_GCC} | sed -e 's|-|/|g')

        if [[ ${MPI_LIBRARY} =~ ^spectrum-mpi-.*$ ]]
        then
            MPI_MODULE=$(echo ${MPI_LIBRARY} | sed -e 's|spectrum-mpi-|spectrum-mpi/|g')
        else
            MPI_MODULE=$(echo ${MPI_LIBRARY} | sed -e 's|-|/|g')
        fi

        # Use the latest CUDA 10, since it's compatible with other
        # CUDA 10.* libraries
        CUDA_MODULE=$(ml --terse avail cuda |& sed -n '/\/10\./p' | tail -n1)

        # Load up the appropriate modules
        module load ${COMPILER_MODULE} ${MPI_MODULE} ${CUDA_MODULE} cmake/3.14.5
        ml save ${SAVELIST_NAME}
    fi

    BRAIN_DIR=/usr/workspace/wsb/brain

    # CUDA-y things (Use the newest)
    ARCH=$(uname -i)
    export NCCL_DIR=$(ls -d --color=no ${BRAIN_DIR}/nccl2/*cuda10*${ARCH} | tail -n1)
    # Right now, we only support cuDNN 7 versions.
    export CUDNN_DIR=$(find ${BRAIN_DIR}/cudnn -maxdepth 2 -type d | grep "cudnn-7.*/cuda-10.*_${ARCH}" | sort -r | head -1)

    # Unit testing framework
    export CLARA_DIR=${WORKSPACE_DIR}/stable_dependencies/clara
    export CATCH2_DIR=${WORKSPACE_DIR}/stable_dependencies/catch2

    # Add Ninja support
    export PATH=${DEPENDENCY_DIR_BASE}/ninja/bin:${PATH}

    # Setup paths to match the build_lbann_lc.sh script (ugh)
    BUILD_DIR_BASE=${LBANN_DIR}/build/gnu.Release.${CLUSTER}.llnl.gov
    BUILD_DIR=${BUILD_DIR_BASE}/lbann/build
    INSTALL_DIR=${BUILD_DIR_BASE}/install

    # Setup a path for Catch2 to use
    CATCH2_OUTPUT_DIR=${LBANN_DIR}/bamboo/compiler_tests
    rm -f ${CATCH2_OUTPUT_DIR}/*.xml

    # Decide if CUDA should be used.
    if [[ "${CLUSTER}" =~ ^(pascal|lassen|ray)$ ]];
    then
        USE_CUDA=ON
    else
        USE_CUDA=OFF
    fi

    # Cleanup
    [[ -e ${BUILD_DIR_BASE} ]] && rm -rf ${BUILD_DIR_BASE}
    mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}

    # Hack to be nice to others.
    if [[ "${CLUSTER}" =~ ^(lassen|ray)$ ]];
    then
        LAUNCH_CMD="lrun -1"
        NHOSTS=$(expr $(printenv LSB_HOSTS | wc -w) - 1)
        NNODES=$(expr ${NHOSTS} / 40)
        PARALLEL_LAUNCH_CMD="jsrun -n${NNODES} -r1 -a4 -c40 -g4 -d packed -b packed:10 "
    else
        unset LAUNCH_CMD
        PARALLEL_LAUNCH_CMD="srun --mpibind=off -N${SLURM_NNODES} --ntasks-per-node=2 "
    fi

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
        -DLBANN_DETERMINISTIC=ON \
        -DLBANN_WARNINGS_AS_ERRORS=ON \
        -DLBANN_WITH_CONDUIT=ON \
        -DLBANN_WITH_CUDA=ON \
        -DLBANN_WITH_NVPROF=OFF \
        -DLBANN_WITH_TBINF=ON \
        -DLBANN_WITH_UNIT_TESTING=ON \
        -DLBANN_WITH_VTUNE=OFF \
        \
        -Dprotobuf_MODULE_COMPATIBLE=ON \
        \
        ${LBANN_DIR} && ${LAUNCH_CMD} ninja && ${LAUNCH_CMD} ninja install && ${LAUNCH_CMD} ./unit_test/seq-catch-tests -r junit -o ${CATCH2_OUTPUT_DIR}/seq_catch_tests_output-${CLUSTER}.xml ; ${PARALLEL_LAUNCH_CMD} ./unit_test/mpi-catch-tests -r junit -o "${CATCH2_OUTPUT_DIR}/mpi_catch_tests_output-${CLUSTER}-rank=%r-size=%s.xml"
else
    ${LBANN_DIR}/scripts/build_lbann_lc.sh --with-conduit
fi
