export MV2_USE_CUDA=1
LBANN_DIR=$(git rev-parse --show-toplevel)
# echo 'Running'
srun ${LBANN_DIR}/build/gnu.Release.pascal.llnl.gov/lbann/build/model_zoo/lbann --reader=${LBANN_DIR}/model_zoo/data_readers/data_reader_imagenet.prototext --model=${LBANN_DIR}/model_zoo/models/densenet/model_densenet_121.prototext --optimizer=${LBANN_DIR}/model_zoo/optimizers/opt_sgd.prototext --learn_rate=0.1 --exit_after_setup >> ${LBANN_DIR}/bamboo/integration_tests/output/densenet_exe_output.txt 2>> ${LBANN_DIR}/bamboo/integration_tests/error/densenet_exe_error.txt
