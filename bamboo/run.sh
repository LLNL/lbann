#!/bin/bash -l

if [ "${CLUSTER}" = 'catalyst' ]; then
    PYTHON=python
fi

if [ "${CLUSTER}" = 'pascal' ]; then
    PYTHON=$bamboo_PYTHON_x86_gpu/python
fi

echo "Task: Cleaning"
./clean.sh

echo "Task: Compiler Tests"
cd compiler_tests
module load cmake/3.9.2
$PYTHON -m pytest -s --junitxml=results.xml
cd ..

echo "Task: Integration Tests"
cd integration_tests
$PYTHON -m pytest -s --junitxml=results.xml
cd ..

echo "Task: Unit Tests"
cd unit_tests
$PYTHON -m pytest -s --junitxml=results.xml
cd ..

echo "Task: Finished"
