export AL_PROGRESS_RANKS_PER_NUMA_NODE=2
export OMP_NUM_THREADS=8
export MV2_USE_RDMA_CM=0
my_hash=spack-build-6xey4bu
my_build=/usr/workspace/wsb/hysom/hdf5_reader/$my_hash

srun --nodes=1 --ntasks=1 --ntasks-per-node=2   --mpibind=off --nvidia_compute_mode=default --cpu_bind=mask_cpu:0x1ff0000001ff,0x3fe0000003fe00 $my_build/model_zoo/lbann --prototext=experiment_train_jag_wae.prototext
