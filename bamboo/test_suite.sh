#!/bin/bash

SUCCESS=0
FAILURE=1

FALSE=0
TRUE=1


EXIT_CODE=${SUCCESS}
FAIL_COUNT=0

function help_message {
    local SCRIPT=$(basename ${0})
    local N=$(tput sgr0)
    local C=$(tput setf 4)
    cat <<EOF
Run test configurations of LBANN.
By default, run all tests.
Usage: ${SCRIPT} [options]
Options:
    ${C}-h|--help${N}                   Display this help message and exit
    ${C}-u|--unit${N}                   Run all unit tests
    ${C}-i|--integration${N}            Run all integration tests 
    ${C}-c|--compiler${N}               Run all compiler tests
    ${C}-e|--executable${N} <val>       Specify path to executable (default executable built from build_lbann_lc.sh)
    ${C}-n|--nobuild${N}                Do not build any executable
EOF
}

function print_label {
    TEST_SET=${1}
    echo "***TEST SET******************************************************************************************************"
    echo "STARTING ${TEST_SET} TESTS"
    echo "*****************************************************************************************************************"
}

function print_results {
    RESULT=$?
    TEST=${1}
    if [ ${RESULT} -ne 0 ]; then
        EXIT_CODE=${FAILURE}
	((FAIL_COUNT++))
	echo "***TEST******************************************************************************************************"
        echo "${TEST} FAILED"
	echo "*************************************************************************************************************"
    else
	echo "***TEST******************************************************************************************************"
        echo "${TEST} PASSED"
        echo "*************************************************************************************************************"
    fi
}

CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
export LBANN_DIR=$(git rev-parse --show-toplevel)
export LBANN_EXE="${LBANN_DIR}/build/${CLUSTER}.llnl.gov/model_zoo/lbann"
export PATH=../../LBANN-NIGHTD-CS/bin:$PATH

# Default is to build the executable from build_lbann_lc.sh
BUILD_EXE=${TRUE}

# Set defaults to run all tests
RUN_ALL=${TRUE}
RUN_UNIT_TESTS=${FALSE}
RUN_INTEGRATION_TESTS=${FALSE}
RUN_COMPILER_TESTS=${FALSE}

while :; do 
    key="$1"
    case $key in
	-h|--help)
	    help_message
	    exit ${SUCCESS}
	    ;;
	-u|--unit)
	    RUN_UNIT_TESTS=${TRUE}
	    RUN_ALL=${FALSE}
	    ;;
	-i|--integration)
	    RUN_INTEGRATION_TESTS=${TRUE}
	    RUN_ALL=${FALSE}
	    ;;
	-c|--compiler)
	    RUN_COMPILER_TESTS=${TRUE}
	    RUN_ALL=${FALSE}
	    # Do NOT set BUILD_EXE=0 because a user could specify -i -c for example.
	    # The -i tests would still need an executable.
	    ;;
	-e|--executable)
	    LBANN_EXE=${2}
	    BUILD_EXE=${FALSE} # User has specified an executable - do not build default.
	    shift
	    ;;
	-n|--nobuild)
	    BUILD_EXE=${FALSE} # User does not want executable to be built
	    ;;
	-?*)
	    echo "Unknown option (${!})" >&2
	    exit ${FAILURE}
	    ;;
	*)
	    break
    esac
    shift
done


source /usr/share/lmod/lmod/init/bash
source /etc/profile.d/00-modulepath.sh
# Build the default executable
if [ ${BUILD_EXE} == ${TRUE} ]
then
    ../scripts/build_lbann_lc.sh --clean-build
fi

# Run the tests
if [ ${RUN_ALL} == ${TRUE} ] || [ ${RUN_UNIT_TESTS} == ${TRUE} ]
then
    print_label "UNIT"

    unit_tests/ridge_regression_test.sh
    print_results "GRADIENT CHECK"

    python -m pytest unit_tests/test_check_proto_models.py
    print_results "PROTOTEXT CHECK"
fi

if [ ${RUN_ALL} == ${TRUE} ] || [ ${RUN_INTEGRATION_TESTS} == ${TRUE} ]
then
    print_label "INTEGRATION"

    python -m pytest integration_tests/accuracy_tests/test_wrapper.py --exe="${LBANN_EXE}"
    print_results "TEST_ACCURACY"

    python -m pytest integration_tests/performance_tests/test_performance.py --exe="${LBANN_EXE}" --dirname="${LBANN_DIR}"
    print_results "TEST_PERFORMANCE"
fi

if [ ${RUN_ALL} == ${TRUE} ] || [ ${RUN_COMPILER_TESTS} == ${TRUE} ]
then
    print_label "COMPILER"

    compiler_tests/test_compiler_variants.sh
    print_results "TEST_COMPILER_VARIANTS"
fi

echo "***Fail Count: ${FAIL_COUNT}***"
exit ${EXIT_CODE}
