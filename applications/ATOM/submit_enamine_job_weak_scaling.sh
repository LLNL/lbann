#!/usr/bin/bash

# this job script studies the relationship between batch size as determined as a quanitity increasing with number of gpu_devices available
# to train one epoch of the validation set of enamine (~40m molecules) 

sbatch --nodes 1 --job-name enamine_1_128_weak enamine_weak_scaling_test.sl 128
sbatch --nodes 2 --job-name enamine_2_128_weak enamine_weak_scaling_test.sl 128
sbatch --nodes 4 --job-name enamine_4_128_weak enamine_weak_scaling_test.sl 128
sbatch --nodes 8 --job-name enamine_8_128_weak enamine_weak_scaling_test.sl 128
sbatch --nodes 16 --job-name enamine_16_128_weak enamine_weak_scaling_test.sl 128
sbatch --nodes 32 --job-name enamine_32_128_weak enamine_weak_scaling_test.sl 128


sbatch --nodes 1 --job-name enamine_1_256_weak enamine_weak_scaling_test.sl 256
sbatch --nodes 2 --job-name enamine_2_256_weak enamine_weak_scaling_test.sl 256 
sbatch --nodes 4 --job-name enamine_4_256_weak enamine_weak_scaling_test.sl 256
sbatch --nodes 8 --job-name enamine_8_256_weak enamine_weak_scaling_test.sl 256
sbatch --nodes 16 --job-name enamine_16_256_weak enamine_weak_scaling_test.sl 256
sbatch --nodes 32 --job-name enamine_32_256_weak enamine_weak_scaling_test.sl 256
