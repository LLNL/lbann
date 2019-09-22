#!/bin/bash

module load mpifileutils

COMPILER=0
while :; do
    case ${1} in
        --compiler)
            # Choose compiler
            if [ -n "${2}" ]; then
                COMPILER=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                exit 1
            fi
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

if [ ${COMPILER} -eq 0 ]; then
    exit 1
fi

LBANN_DIR=$(git rev-parse --show-toplevel)
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
FILE_PREFIX=${LBANN_DIR}/bamboo/unit_tests/output/full_alexnet_${CLUSTER}_${COMPILER}

# Clear SSDs
srun --wait=0 --clear-ssd hostname > ${FILE_PREFIX}_1_output.txt

# Cache dataset
echo "Caching dataset..."
[ -e /l/ssd/lbannusr/datasets-resized/ILSVRC2012/train_resized.tar ] || \
  srun --nodes=128 --ntasks-per-node=2 dbcast /p/lscratchh/brainusr/datasets/ILSVRC2012/original/train_resized.tar /l/ssd/lbannusr/datasets-resized/ILSVRC2012/train_resized.tar > ${FILE_PREFIX}_2_output.txt
[ -d /l/ssd/lbannusr/datasets-resized/ILSVRC2012/train ] || \
  srun --nodes=128 --ntasks-per-node=1 tar xf /l/ssd/lbannusr/datasets-resized/ILSVRC2012/train_resized.tar -C /l/ssd/lbannusr/datasets-resized/ILSVRC2012
[ -e /l/ssd/lbannusr/datasets-resized/ILSVRC2012/val_resized.tar ] || \
  srun --nodes=128 --ntasks-per-node=2 dbcast /p/lscratchh/brainusr/datasets/ILSVRC2012/original/val_resized.tar /l/ssd/lbannusr/datasets-resized/ILSVRC2012/val_resized.tar > ${FILE_PREFIX}_3_output.txt
[ -d /l/ssd/lbannusr/datasets-resized/ILSVRC2012/val ] || \
  srun --nodes=128 --ntasks-per-node=1 tar xf /l/ssd/lbannusr/datasets-resized/ILSVRC2012/val_resized.tar -C /l/ssd/lbannusr/datasets-resized/ILSVRC2012
[ -e /l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels.tar ] || \
  srun --nodes=128 --ntasks-per-node=2 dbcast /p/lscratchh/brainusr/datasets/ILSVRC2012/original/labels.tar /l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels.tar > ${FILE_PREFIX}_4_output.txt
[ -e /l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels/train.txt ] || \
  srun --nodes=128 --ntasks-per-node=1 tar xf /l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels.tar -C /l/ssd/lbannusr/datasets-resized/ILSVRC2012
wait
echo "Done caching dataset..."

# Experiment
srun --nodes=128 --ntasks-per-node=2 ${LBANN_DIR}/bamboo/compiler_tests/builds/catalyst_gcc-7.1.0_x86_64_mvapich2-2.2_openblas_rel/build/model_zoo/lbann --model=${LBANN_DIR}/model_zoo/models/alexnet/model_alexnet.prototext --optimizer=${LBANN_DIR}/model_zoo/optimizers/opt_sgd.prototext --reader=${LBANN_DIR}/model_zoo/data_readers/data_reader_imagenet.prototext --data_filedir_train=/l/ssd/lbannusr/datasets-resized/ILSVRC2012/train/ --data_filename_train=/l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels/train.txt --data_filedir_test=/l/ssd/lbannusr/datasets-resized/ILSVRC2012/val/ --data_filename_test=/l/ssd/lbannusr/datasets-resized/ILSVRC2012/labels/val.txt
