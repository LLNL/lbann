#!/bin/bash -l

CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
LBANN_DIR=$(git rev-parse --show-toplevel)

cd ${LBANN_DIR}/ci_test

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Running Integration and Unit tests"
echo "~~~~~ lbann: $(which lbann)"
echo "~~~~~ $(date)"
echo "----- PATH: ${PATH}"
echo "----- lbann_pfe.sh: $(which lbann_pfe.sh)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

PYTHON=python3
LBANN_PYTHON=lbann_pfe.sh

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

export OMP_NUM_THREADS=10

# These tests are "allowed" to fail inside the script. That is, the
# unit tests should be run even if these fail. The status is cached
# for now.
status=0
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Task: Integration Tests with file pattern: ${TEST_FLAG}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd integration_tests
$LBANN_PYTHON -m pytest -vv --durations=0 --junitxml=${LBANN_DIR}/integration_test_results_junit.xml ${TEST_FLAG} || {
    this_status=$?
    status=$(( $status + $this_status ))
    failed_tests=$(( $failed_tests + $this_status ))
    echo "******************************"
    echo " >>> Integration Tests FAILED"
    echo "******************************"
}
cd ..

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Task: Unit Tests with file pattern: ${TEST_FLAG}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
cd unit_tests
$LBANN_PYTHON -m pytest -vv --durations=0 --junitxml=${LBANN_DIR}/unit_test_results_junit.xml ${TEST_FLAG} || {
    this_status=$?
    status=$(( $status + $this_status ))
    failed_tests=$(( $failed_tests + $this_status ))
    echo "******************************"
    echo " >>> Unit Tests FAILED"
    echo "******************************"
}
cd ..

echo "Task: Finished with status ${status} and ${failed_tests} failed tests"
