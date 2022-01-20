#!/bin/bash

# Just in case
source ${HOME}/spack_repos/spack_${SYSTEM_NAME}.git/share/spack/setup-env.sh

# Load up the spack environment
SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)
spack env activate lbann-${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}
spack load lbann@${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET} arch=${SPACK_ARCH}

FAILED_JOBS=""

SPACK_BUILD_DIR=$(find . -iname "spack-build-*" -type d | head -n 1)
cd ${SPACK_BUILD_DIR}
srun --jobid=${JOB_ID} -N 1 -n 1 -t 5 \
     ./unit_test/seq-catch-tests \
     -r JUnit \
     -o seq-catch-results.xml
if [[ $? -ne 0 ]]; then
    FAILED_JOBS+=" seq"
fi

srun --jobid=${JOB_ID} \
     -N 2 -n $(($TEST_TASKS_PER_NODE * 2)) \
     --ntasks-per-node=$TEST_TASKS_PER_NODE \
     -t 5 ${TEST_MPIBIND_FLAG} \
     ./unit_test/mpi-catch-tests \
     -r JUnit \
     -o "mpi-catch-results-rank=%r-size=%s.xml"
if [[ $? -ne 0 ]]; then
    FAILED_JOBS+=" mpi"
fi

srun --jobid=${JOB_ID} \
     -N 2 -n $(($TEST_TASKS_PER_NODE * 2)) \
     --ntasks-per-node=$TEST_TASKS_PER_NODE \
     -t 5 ${TEST_MPIBIND_FLAG} \
     ./unit_test/mpi-catch-tests "[filesystem]" \
     -r JUnit \
     -o "mpi-catch-filesystem-results-rank=%r-size=%s.xml"
if [[ $? -ne 0 ]];
then
    FAILED_JOBS+=" mpi-filesystem"
fi

# Try to write a semi-useful message to this file since it's being
# saved as an artifact. It's not completely outside the realm that
# someone would look at it.
if [[ -n "${FAILED_JOBS}" ]];
then
    echo "Some Catch2 tests failed:${FAILED_JOBS}" > catch-tests-failed.txt
fi

# Return "success" so that the pytest-based testing can run.
exit 0
