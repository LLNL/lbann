#!/bin/sh

# Get LBANN parameters
# Note: We assume this script is executed on an LLNL LC system with a
# working directory within an LBANN project directory.
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
LBANN_DIR=$(git rev-parse --show-toplevel)
LBANN_EXE="${LBANN_DIR}/build/${CLUSTER}.llnl.gov/model_zoo/lbann"

# Run experiment
salloc -N1 srun --ntasks=1 ${LBANN_EXE} --model=${LBANN_DIR}/model_zoo/tests/model_mnist_ridge_regression.prototext --optimizer=${LBANN_DIR}/model_zoo/optimizers/opt_adam.prototext --reader=${LBANN_DIR}/model_zoo/data_readers/data_reader_mnist.prototext

# Return status
if [ $? -ne 0 ]; then
    echo "---------------------"
    echo "GRADIENT CHECK FAILED"
    echo "---------------------"
    exit 1
else
    echo "---------------------"
    echo "GRADIENT CHECK PASSED"
    echo "---------------------"
    exit 0
fi
