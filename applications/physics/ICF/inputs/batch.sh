#!/bin/bash
#SBATCH --chdir=/usr/WS2/hysom/hdf5_reader/bamboo/integration_tests/experiments/test_integration_lenet
#SBATCH --output=/usr/WS2/hysom/hdf5_reader/bamboo/integration_tests/experiments/test_integration_lenet/out.log
#SBATCH --error=/usr/WS2/hysom/hdf5_reader/bamboo/integration_tests/experiments/test_integration_lenet/err.log
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --job-name=lbann_test_integration_lenet
#SBATCH --mpibind=off
#SBATCH --nvidia_compute_mode=default
#SBATCH --cpu_bind=mask_cpu:0x1ff0000001ff,0x3fe0000003fe00

export AL_PROGRESS_RANKS_PER_NUMA_NODE=2
export OMP_NUM_THREADS=8
export MV2_USE_RDMA_CM=0
echo "Started at $(date)"
srun --mpibind=off --nvidia_compute_mode=default --cpu_bind=mask_cpu:0x1ff0000001ff,0x3fe0000003fe00 --chdir=/usr/WS2/hysom/hdf5_reader/bamboo/integration_tests/experiments/test_integration_lenet --nodes=2 --ntasks=4 --ntasks-per-node=2 --job-name=lbann_test_integration_lenet /usr/WS2/hysom/spack/opt/spack/linux-rhel7-broadwell/gcc-8.3.1/lbann-local-broadwell-6xey4bumsdsplfqjywknp5tx2cqrtwd6/bin/lbann --prototext=/usr/WS2/hysom/hdf5_reader/bamboo/integration_tests/experiments/test_integration_lenet/experiment.prototext
status=$?
echo "Finished at $(date)"
exit ${status}
