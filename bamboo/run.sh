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

echo "run.sh WEEKLY="
echo $WEEKLY

echo "Task: Cleaning"
./clean.sh

echo "Task: Compiler Tests"
cd compiler_tests
module load cmake/3.9.2
$PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml
cd ..

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
$PYTHON -m pytest -s -vv --durations=0 --junitxml=results.xml
cd ..

echo "Task: Finished"
