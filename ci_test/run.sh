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

source ${LBANN_DIR}/scripts/utilities.sh

echo "run.sh WEEKLY="
echo $WEEKLY

echo "Task: Cleaning"
./clean.sh

echo "Discovered installed module file: ${LBANN_MODFILES_DIR}"
echo "Task: Compiler Tests"
cd compiler_tests
$PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml || exit 1

# Find the correct module to load
ml use ${LBANN_MODFILES_DIR}
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
