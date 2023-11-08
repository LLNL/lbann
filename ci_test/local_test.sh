#!/bin/bash -l

# Local testing (i.e. not with Bamboo)

################################################################
# Help message
################################################################

function help_message {
    local SCRIPT=$(basename ${0})
    local N=$(tput sgr0)    # Normal text
    local C=$(tput setf 4)  # Colored text
    cat << EOF
Run integration and unit tests locally, outside Bamboo.
Usage: ./${SCRIPT} [options]
Options:
  ${C}--help${N}                      Display this help message and exit.
  ${C}--data-reader-fraction${N} <val> Specify data reader fraction.
  ${C}--integration-tests${N}         Specify that only integration tests should be run.
  ${C}--unit-tests${N}                Specify that only unit tests should be run.
EOF
}

################################################################
# Parse command-line arguments
################################################################

DATA_READER_FRACTION=0.001
INTEGRATION_TESTS=1
UNIT_TESTS=1
while :; do
    case ${1} in
        -h|--help)
            # Help message
            help_message
            exit 0
            ;;
        -d|--data-reader-fraction)
            # Set data reader fraction.
            # -n: check if string has non-zero length.
            if [ -n "${2}" ]; then
                DATA_READER_FRACTION=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                help_message
                exit 1
            fi
            ;;
        -i|--integration-tests)
            # Run only integration tests
            UNIT_TESTS=0
            ;;
        -u|--unit-tests)
            # Run only unit tests
            INTEGRATION_TESTS=0
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

################################################################
# Run tests
################################################################

# Assume user already has an executable (i.e. no need for compiler tests).
# Assume user already has 16 nodes allocated on a cluster.

echo "EXECUTABLE=${EXECUTABLE}"
echo "INTEGRATION_TESTS=${INTEGRATION_TESTS}"
echo "UNIT_TESTS=${UNIT_TESTS}"
PYTHON=python3

echo "Task: Cleaning"
./clean.sh

echo "Task: Integration Tests"
cd integration_tests
if [ ${INTEGRATION_TESTS} -ne 0 ]; then
    $PYTHON -m pytest -s -vv --durations=0
fi
cd ..

echo "Task: Unit Tests"
cd unit_tests
if [ ${UNIT_TESTS} -ne 0 ]; then
    $PYTHON -m pytest -s -vv --durations=0 --data-reader-fraction=${DATA_READER_FRACTION}
fi
cd ..

echo "Task: Finished"
