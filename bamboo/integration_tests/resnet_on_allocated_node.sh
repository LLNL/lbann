export MV2_USE_CUDA=1
LBANN_DIR=$(git rev-parse --show-toplevel)

# Replace 18 with 101 or 152 to test other ResNet versions.

# Append to output (use if checkpointing)
#srun --ntasks-per-node=2 ${LBANN_DIR}/build/gnu.Release.pascal.llnl.gov/install/bin/lbann --model=${LBANN_DIR}/model_zoo/models/resnet/resnet18/model_resnet18_sequential.prototext --reader=${LBANN_DIR}/model_zoo/data_readers/data_reader_imagenet.prototext --optimizer=${LBANN_DIR}/model_zoo/optimizers/opt_sgd.prototext --learn_rate=0.1 --data_filedir_train="/p/lscratchh/brainusr/datasets/ILSVRC2012/original/train/" --data_filename_train="/p/lscratchh/brainusr/datasets/ILSVRC2012/origin" --data_filedir_test="/p/lscratchh/brainusr/datasets/ILSVRC2012/original/val/" --data_filename_test="/p/lscratchh/brainusr/datasets/ILSVRC2012/original/labels/val.txt" >> ${LBANN_DIR}/bamboo/integration_tests/output/resnet_exe_output.txt 2>> ${LBANN_DIR}/bamboo/integration_tests/error/resnet_exe_error.txt

# Append to old output, assume data reader uses l-scratch-h
srun --ntasks-per-node=2 ${LBANN_DIR}/build/gnu.Release.pascal.llnl.gov/install/bin/lbann --model=${LBANN_DIR}/model_zoo/models/resnet/resnet18/model_resnet18_sequential.prototext --reader=${LBANN_DIR}/model_zoo/data_readers/data_reader_imagenet.prototext --optimizer=${LBANN_DIR}/model_zoo/optimizers/opt_sgd.prototext --learn_rate=0.1 >> ${LBANN_DIR}/bamboo/integration_tests/output/resnet_exe_output.txt 2>> ${LBANN_DIR}/bamboo/integration_tests/error/resnet_exe_error.txt

# Write over old output & exit after setup
#srun --ntasks-per-node=2 ${LBANN_DIR}/build/gnu.Release.pascal.llnl.gov/install/bin/lbann --model=${LBANN_DIR}/model_zoo/models/resnet/resnet18/model_resnet18_sequential.prototext --reader=${LBANN_DIR}/model_zoo/data_readers/data_reader_imagenet.prototext --optimizer=${LBANN_DIR}/model_zoo/optimizers/opt_sgd.prototext --learn_rate=0.1 --data_filedir_train="/p/lscratchh/brainusr/datasets/ILSVRC2012/original/train/" --data_filename_train="/p/lscratchh/brainusr/datasets/ILSVRC2012/origin" --data_filedir_test="/p/lscratchh/brainusr/datasets/ILSVRC2012/original/val/" --data_filename_test="/p/lscratchh/brainusr/datasets/ILSVRC2012/original/labels/val.txt" --exit_after_setup > ${LBANN_DIR}/bamboo/integration_tests/output/resnet_exe_output.txt 2> ${LBANN_DIR}/bamboo/integration_tests/error/resnet_exe_error.txt

# Write over old output & exit after setup, assume data reader uses l-scratch-h
#srun --ntasks-per-node=2 ${LBANN_DIR}/build/gnu.Release.pascal.llnl.gov/install/bin/lbann --model=${LBANN_DIR}/model_zoo/models/resnet/resnet18/model_resnet18_sequential.prototext --reader=${LBANN_DIR}/model_zoo/data_readers/data_reader_imagenet.prototext --optimizer=${LBANN_DIR}/model_zoo/optimizers/opt_sgd.prototext --learn_rate=0.1 --exit_after_setup > ${LBANN_DIR}/bamboo/integration_tests/output/resnet_exe_output.txt 2> ${LBANN_DIR}/bamboo/integration_tests/error/resnet_exe_error.txt
