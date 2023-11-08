#!/bin/bash
#SBATCH --nodes 16
#SBATCH --partition pbatch
#SBATCH --time 1440

# Update "--time" above to increase/decrease allocation time.
# Update "executable" with your executable.
# Use "data-reader-fraction" to specify data reader fraction.
# Use "--integration-tests" to only run integration tests.
# Use "--unit-tests" to only run unit tests.
./local_test.sh --executable "../build/gnu.Release.pascal.llnl.gov/install/bin/lbann" --data-reader-fraction 0.001 --unit-tests
