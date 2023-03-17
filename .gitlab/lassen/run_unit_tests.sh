## Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
# Just in case
source ${HOME}/${SPACK_REPO}/share/spack/setup-env.sh
source spack-ci-env-name.sh

# Load up the spack environment
spack env activate lbann-${SPACK_DEP_ENV_NAME}-${SPACK_ARCH_TARGET}
spack load lbann@${SPACK_DEP_ENV_NAME}-${SPACK_ARCH_TARGET} arch=${SPACK_ARCH}

echo "Task: Unit Tests"
cd ci_test/unit_tests
export OMP_NUM_THREADS=10
lrun -N ${LBANN_NNODES} -T $TEST_TASKS_PER_NODE python3 -m pytest -s -vv --durations=0 --junitxml=results.xml
status=$(($status + $?))

echo "Task: Finished"
exit $status
