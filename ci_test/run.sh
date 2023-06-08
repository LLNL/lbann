#!/bin/bash -l

CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
LBANN_DIR=$(git rev-parse --show-toplevel)

cd ${LBANN_DIR}/ci_test

echo "${PWD}/run.sh CLUSTER=${CLUSTER}"

PYTHON=python3
LBANN_PYTHON=lbann_pfe.sh

WEEKLY=0
while :; do
    case ${1} in
        --weekly)
            # Run all tests. This is a weekly build.
            echo "Setting WEEKLY in run.sh"
            WEEKLY=1
            ;;
        -?*)
            # Unknown option
            echo "Unknown option (${1})" >&2
            exit 1
            ;;
        *)
            # Break loop if there are no more options
            break
    esac
    shift
done

# Use the spack provided by the CI
source ${HOME}/${SPACK_REPO}/share/spack/setup-env.sh

# "spack" is just a shell function; it may not be exported to this
# scope. Just to be sure, reload the shell integration.
if [ -n "${SPACK_ROOT}" ]; then
    source ${SPACK_ROOT}/share/spack/setup-env.sh
else
    echo "Spack required.  Please set SPACK_ROOT environment variable"
    exit 1
fi

SPACK_VERSION=$(spack --version | sed 's/-.*//g' | sed 's/[(].*[)]//g')
MIN_SPACK_VERSION=0.18.0

source ${LBANN_DIR}/scripts/utilities.sh

compare_versions ${SPACK_VERSION} ${MIN_SPACK_VERSION}
VALID_SPACK=$?

if [[ ${VALID_SPACK} -eq 2 ]]; then
    echo "Newer version of Spack required.  Detected version ${SPACK_VERSION} requires at least ${MIN_SPACK_VERSION}"
    exit 1
fi

echo "run.sh WEEKLY="
echo $WEEKLY

echo "Task: Cleaning"
./clean.sh

echo "I think that the environment is ${SPACK_ENV_NAME}"
echo "Task: Compiler Tests"
cd compiler_tests
$PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml || exit 1

# Find the correct module to load
SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)
SPACK_ENV_CMD="spack env activate -p lbann-${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}"
echo ${SPACK_ENV_CMD} | tee -a ${LOG}
${SPACK_ENV_CMD}
echo "source ${LBANN_DIR}/LBANN_${SYSTEM_NAME}_${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}_setup_module_path.sh" | tee -a ${LOG}
source ${LBANN_DIR}/LBANN_${SYSTEM_NAME}_${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}_setup_module_path.sh
ml load lbann

echo "Testing $(which lbann)"
cd ..

# These tests are "allowed" to fail inside the script. That is, the
# unit tests should be run even if these fail. The status is cached
# for now.
echo "Task: Integration Tests"
cd integration_tests
if [ ${WEEKLY} -ne 0 ]; then
    $LBANN_PYTHON -m pytest -s -vv --durations=0 --weekly --junitxml=results.xml
    status=$?
else
    $LBANN_PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml
    status=$?
fi
cd ..

echo "Task: Unit Tests"
cd unit_tests
OMP_NUM_THREADS=10 $LBANN_PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml
status=$(($status + $?))
cd ..

echo "Task: Finished"
exit $status
