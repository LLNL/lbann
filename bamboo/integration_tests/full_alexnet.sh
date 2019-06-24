#!/bin/bash

module load mpifileutils

# Clear SSDs
srun --wait=0 --clear-ssd hostname > /dev/null

# Cache dataset
echo "Caching dataset..."
[ -e /l/ssd/lbannusr/datasets-resized/ILSVRC2012/train_resized.tar ] || \
  srun --nodes=128 --ntasks-per-node=2 dbcast /p/lscratchh/brainusr/datasets/ILSVRC2012/original/train_resized.tar /l/ssd/lbannusr/datasets-resized/ILSVRC2012/train_resized.tar > /dev/null
[ -d /l/ssd/lbannusr/datasets-resized/ILSVRC2012/train ] || \
  srun --nodes=128 --ntasks-per-node=1 tar xf /l/ssd/lbannusr/datasets-resized/ILSVRC2012/train_resized.tar -C /l/ssd/lbannusr/datasets-resized/ILSVRC2012
[ -e /l/ssd/lbannusr/datasets-resized/ILSVRC2012/val_resized.tar ] || \
  srun --nodes=128 --ntasks-per-node=2 dbcast /p/lscratchh/brainusr/datasets/ILSVRC2012/original/val_resized.tar /l/ssd/lbannusr/datasets-resized/ILSVRC2012/val_resized.tar > /dev/null
[ -d /l/ssd/lbannusr/datasets-resized/ILSVRC2012/val ] || \
  srun --nodes=128 --ntasks-per-node=1 tar xf /l/ssd/lbannusr/datasets-resized/ILSVRC2012/val_resized.tar -C /l/ssd/lbannusr/datasets-resized/ILSVRC2012
[ -e /l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels.tar ] || \
  srun --nodes=128 --ntasks-per-node=2 dbcast /p/lscratchh/brainusr/datasets/ILSVRC2012/original/labels.tar /l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels.tar > /dev/null
[ -e /l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels/train.txt ] || \
  srun --nodes=128 --ntasks-per-node=1 tar xf /l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels.tar -C /l/ssd/lbannusr/datasets-resized/ILSVRC2012
wait
echo "Done caching dataset..."

LBANN_DIR=$(git rev-parse --show-toplevel)
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

# Experiment
srun --nodes=128 --ntasks-per-node=2 ${LBANN_DIR}/bamboo/compiler_tests/builds/catalyst_gcc-7.1.0_x86_64_mvapich2-2.2_openblas_rel/build/model_zoo/lbann --model=${LBANN_DIR}/model_zoo/models/alexnet/model_alexnet.prototext --optimizer=${LBANN_DIR}/model_zoo/optimizers/opt_sgd.prototext --reader=${LBANN_DIR}/model_zoo/data_readers/data_reader_imagenet.prototext --data_filedir_train=/l/ssd/lbannusr/datasets-resized/ILSVRC2012/train/ --data_filename_train=/l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels/train.txt --data_filedir_test=/l/ssd/lbannusr/datasets-resized/ILSVRC2012/val/ --data_filename_test=/l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels/val.txt
