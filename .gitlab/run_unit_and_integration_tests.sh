#!/bin/bash -l

CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
LBANN_DIR=$(git rev-parse --show-toplevel)

cd ${LBANN_DIR}/ci_test

echo "${PWD}/run.sh CLUSTER=${CLUSTER}"

PYTHON=python3
LBANN_PYTHON=lbann_pfe.sh

# WEEKLY=0
# while :; do
#     case ${1} in
#         --weekly)
#             # Run all tests. This is a weekly build.
#             echo "Setting WEEKLY in run.sh"
#             WEEKLY=1
#             ;;
#         -?*)
#             # Unknown option
#             echo "Unknown option (${1})" >&2
#             exit 1
#             ;;
#         *)
#             # Break loop if there are no more options
#             break
#     esac
#     shift
# done

# # Use the spack provided by the CI
# source ${HOME}/${SPACK_REPO}/share/spack/setup-env.sh

# # "spack" is just a shell function; it may not be exported to this
# # scope. Just to be sure, reload the shell integration.
# if [ -n "${SPACK_ROOT}" ]; then
#     source ${SPACK_ROOT}/share/spack/setup-env.sh
# else
#     echo "Spack required.  Please set SPACK_ROOT environment variable"
#     exit 1
# fi

# SPACK_VERSION=$(spack --version | sed 's/-.*//g' | sed 's/[(].*[)]//g')
# MIN_SPACK_VERSION=0.18.0

# source ${LBANN_DIR}/scripts/utilities.sh

# compare_versions ${SPACK_VERSION} ${MIN_SPACK_VERSION}
# VALID_SPACK=$?

# if [[ ${VALID_SPACK} -eq 2 ]]; then
#     echo "Newer version of Spack required.  Detected version ${SPACK_VERSION} requires at least ${MIN_SPACK_VERSION}"
#     exit 1
# fi

# echo "run.sh WEEKLY="
# echo $WEEKLY

# echo "Task: Cleaning"
# ./clean.sh

# echo "Discovered installed module file: ${LBANN_MODFILES_DIR}"
# echo "Discovered Spack environment: ${SPACK_ENV_NAME}"
# echo "Task: Compiler Tests"
# cd compiler_tests
# $PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml || exit 1

# Find the correct module to load
# SPACK_ARCH=$(spack arch)
# SPACK_ARCH_TARGET=$(spack arch -t)
# export LBANN_BUILD_LABEL="lbann_${SYSTEM_NAME}_${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}"
# export LBANN_BUILD_PARENT_DIR="${CI_PROJECT_DIR}/builds/${LBANN_BUILD_LABEL}"
# export LBANN_INSTALL_DIR="${LBANN_BUILD_PARENT_DIR}/install"
# export LBANN_MODFILES_DIR="${LBANN_INSTALL_DIR}/etc/modulefiles"
# ml use ${LBANN_MODFILES_DIR}
# ml load lbann

# cd unit_tests
# echo "Testing $(which lbann) from $(pwd)"

case "${cluster}" in
    pascal)
        export OMPI_MCA_mpi_warn_on_fork=0
        ;;
    lassen)
        ;;
    corona|tioga)
        export H2_SELECT_DEVICE_0=1
        ;;
    *)
        echo "Unknown cluster: ${cluster}"
        ;;
esac

# These tests are "allowed" to fail inside the script. That is, the
# unit tests should be run even if these fail. The status is cached
# for now.
echo "Task: Integration Tests"
cd integration_tests
$LBANN_PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml
status=$?
cd ..

echo "Task: Unit Tests"
cd unit_tests
$LBANN_PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml
status=$(($status + $?))
cd ..

echo "Task: Finished"
exit $status
