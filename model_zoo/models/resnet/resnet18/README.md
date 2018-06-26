# LBANN Resnet-18
- Source: https://arxiv.org/abs/1512.03385

- run

  - srun --nodes=8 --ntasks-per-node=2 build/gnu.Release.pascal.llnl.gov/install/bin/lbann --model=model_zoo/models/resnet18/model_resnet18_sequential.prototext --reader=model_zoo/data_readers/data_reader_imagenet.prototext --optimizer=model_zoo/optimizers/opt_sgd.prototext

- Accuracty

  | Framework | Top-1 error | Top-5 error |
  | ---       | ---         | ---         |
  | Pytorch   | 30.04       | 10.92          |
