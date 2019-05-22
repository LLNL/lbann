#!/bin/bash -l

echo "Task: Cleaning"
./clean.sh

echo "Task: Compiler Tests"
cd compiler_tests
module load cmake/3.9.2
python -m pytest -s --junitxml=results.xml
cd ..

echo "Task: Integration Tests"
cd integration_tests
python -m pytest -s --junitxml=results.xml
cd ..

echo "Task: Unit Tests"
cd unit_tests
python -m pytest -s --junitxml=results.xml
cd ..

echo "Task: Finished"
