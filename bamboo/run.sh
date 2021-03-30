#!/bin/bash -l

CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

echo "run.sh CLUSTER="
echo $CLUSTER

PYTHON=python3

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

# "spack" is just a shell function; it may not be exported to this
# scope. Just to be sure, reload the shell integration.
if [ -n "${SPACK_ROOT}" ]; then
    source ${SPACK_ROOT}/share/spack/setup-env.sh
else
    echo "Spack required.  Please set SPACK_ROOT environment variable"
    exit 1
fi

SPACK_VERSION=$(spack --version | sed 's/-.*//g')
MIN_SPACK_VERSION=0.16.0

LBANN_DIR=$(git rev-parse --show-toplevel)
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

echo "Task: Compiler Tests"
cd compiler_tests
$PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml
# Find the correct module to load
SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)
SPACK_LOAD_CMD="spack load lbann@bamboo-${SPACK_ARCH_TARGET} arch=${SPACK_ARCH}"
echo ${SPACK_LOAD_CMD} | tee -a ${LOG}
${SPACK_LOAD_CMD}
echo $(which lbann)
# LBANN_FIND_CMD="spack find --format {hash:7} lbann@bamboo-${SPACK_ARCH_TARGET} arch=${SPACK_ARCH}"
# echo ${LBANN_FIND_CMD} | tee -a ${LOG}
# LBANN_HASH=$(${LBANN_FIND_CMD})
# if [[ -n "${LBANN_HASH}" && ! "${LBANN_HASH}" =~ "No package matches the query" ]]; then
#     LBANN_HASH_ARRAY=(${LBANN_HASH})
#     echo "Our array length is ${#LBANN_HASH_ARRAY[@]}"
#     if [[ ${#LBANN_HASH_ARRAY[@]} -ne 1 ]]; then
#         echo "Unable to find a single LBANN executable"
#         exit 1
#     fi
#     for h in ${LBANN_HASH_ARRAY[@]}
#     do
#         CMD="module load lbann/bamboo-${SPACK_ARCH_TARGET}-${h}"
#         echo ${CMD} | tee -a ${LOG}
#         [[ -z "${DRY_RUN:-}" ]] && ${CMD}
#     done
# else
#     echo "Unable to find the LBANN executable in Spack"
#     exit 1
# fi
cd ..
#exit

echo "Task: Integration Tests"
cd integration_tests
if [ ${WEEKLY} -ne 0 ]; then
    $PYTHON -m pytest -s -vv --durations=0 --weekly --junitxml=results.xml
else
    $PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml
fi
cd ..

echo "Task: Unit Tests"
cd unit_tests
OMP_NUM_THREADS=10 $PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml
cd ..

echo "Task: Finished"
