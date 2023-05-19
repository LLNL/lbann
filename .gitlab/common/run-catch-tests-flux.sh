################################################################################
## Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#!/bin/bash

# This script needs to be run under a flux proxy ${JOB_ID} command
# Just in case
source ${HOME}/${SPACK_REPO}/share/spack/setup-env.sh

# Load the LBANN module
echo "BVE Starting catch tests from ${PWD}"
echo "${LBANN_MODFILES_DIR}"
#echo "source LBANN_${SYSTEM_NAME}_${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}_setup_module_path.sh"
file LBANN_${SYSTEM_NAME}_${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}_setup_module_path.sh
file ${PWD}/LBANN_${SYSTEM_NAME}_${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}_setup_module_path.sh
#source LBANN_${SYSTEM_NAME}_${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}_setup_module_path.sh
echo "$(which lbann)"
ml load lbann
echo "$(which lbann)"

# Load up the spack environment
#spack env activate -p lbann-${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}
#spack load lbann@${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET} arch=${SPACK_ARCH}

# Configure the output directory
OUTPUT_DIR=${CI_PROJECT_DIR}/${RESULTS_DIR}
if [[ -d ${OUTPUT_DIR} ]];
then
    rm -rf ${OUTPUT_DIR}
fi
mkdir -p ${OUTPUT_DIR}

FAILED_JOBS=""
export MV2_USE_RDMA_CM=0
#export OMPI_MCA_btl=^openib
#export OMPI_MCA_osc=ucx

# ml
# module load gcc-tce/10.3.1 rocm/5.2.0 openmpi-tce/4.1.2
# ml

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}

# LBANN_HASH=$(spack find --format {hash:7} lbann@${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET})
# SPACK_BUILD_DIR="spack-build-${LBANN_HASH}"

# cd ${SPACK_BUILD_DIR}

cd ${LBANN_BUILD_DIR}

flux resource list
#flux proxy ${JOB_ID} flux resource list

#flux mini run -N1 -n1 env

flux run --label-io -n4 -N2 -g 1 -o cpu-affinity=per-task -o gpu-affinity=per-task sh -c 'taskset -cp $$; printenv | grep VISIBLE' | sort

flux run --label-io -n4 -N2 -g 1 -o cpu-affinity=off -o gpu-affinity=per-task sh -c 'taskset -cp $$; printenv | grep VISIBLE' | sort
#flux proxy ${JOB_ID} flux mini run --label-io -n4 -N2 -g 1 -o cpu-affinity=per-task -o gpu-affinity=per-task sh -c 'taskset -cp $$; printenv | grep VISIBLE' | sort

flux run -N 1 -n 1 -g 1 -t 5m rocm-smi

     # module load gcc-tce/10.3.1 rocm/5.2.0 openmpi-tce/4.1.2; \
     # source /g/g14/lbannusr/spack_repos/spack_corona.git/share/spack/setup-env.sh; \
     # spack env activate -p lbann-${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}; \
#flux proxy ${JOB_ID} flux mini run -N 1 -n 1 -g 1 -t 5m \
flux run -N 1 -n 1 -g 1 -t 5m \
     ./unit_test/seq-catch-tests \
     -r JUnit \
     -o ${OUTPUT_DIR}/seq-catch-results.xml
if [[ $? -ne 0 ]]; then
    FAILED_JOBS+=" seq"
fi

#     --ntasks-per-node=$TEST_TASKS_PER_NODE \
# ${TEST_MPIBIND_FLAG}
    #LBANN_NNODES=1
#TEST_TASKS_PER_NODE=4
#flux proxy ${JOB_ID} flux mini run \
flux run \
     -N ${LBANN_NNODES} -n $((${TEST_TASKS_PER_NODE} * ${LBANN_NNODES})) \
     -g 1 -t 5m -o gpu-affinity=per-task -o cpu-affinity=per-task -o mpibind=off \
     ./unit_test/mpi-catch-tests "exclude:[random]" "exclude:[filesystem]"\
     -r JUnit \
     -o "${OUTPUT_DIR}/mpi-catch-results-rank=%r-size=%s.xml"
if [[ $? -ne 0 ]]; then
    FAILED_JOBS+=" mpi"
fi

#     --ntasks-per-node=$TEST_TASKS_PER_NODE \
# ${TEST_MPIBIND_FLAG}
#flux proxy ${JOB_ID} flux mini run \
flux run \
     -N ${LBANN_NNODES} -n $((${TEST_TASKS_PER_NODE} * ${LBANN_NNODES})) \
     -g 1 -t 5m -o gpu-affinity=per-task -o cpu-affinity=per-task -o mpibind=off \
     ./unit_test/mpi-catch-tests -s "[filesystem]" \
     -r JUnit \
     -o "${OUTPUT_DIR}/mpi-catch-filesystem-results-rank=%r-size=%s.xml"
if [[ $? -ne 0 ]];
then
    FAILED_JOBS+=" mpi-filesystem"
fi

# Try to write a semi-useful message to this file since it's being
# saved as an artifact. It's not completely outside the realm that
# someone would look at it.
if [[ -n "${FAILED_JOBS}" ]];
then
    echo "Some Catch2 tests failed:${FAILED_JOBS}" > ${OUTPUT_DIR}/catch-tests-failed.txt
fi

echo "I have found the following results >>>"
cat ${OUTPUT_DIR}/catch-tests-failed.txt
echo "<<< EOL"

# Return "success" so that the pytest-based testing can run.
exit 0
