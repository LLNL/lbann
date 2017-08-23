example invocation:
 $ srun  --nodes=8 --ntasks-per-node=2  --nvidia_compute_mode=default ./../../build/surface.llnl.gov/model_zoo/lbann --model=model_resnet50.prototext --reader=../prototext/data_reader_imagenet.prototext --optimizer=../prototext/opt_adam.prototext
