#!/bin/sh

# Run gradiant check
salloc -N1 srun --ntasks=1 ${LBANN_EXE} --model=${LBANN_DIR}/model_zoo/tests/model_mnist_ridge_regression.prototext --optimizer=${LBANN_DIR}/model_zoo/optimizers/opt_adam.prototext --reader=${LBANN_DIR}/model_zoo/data_readers/data_reader_mnist.prototext
