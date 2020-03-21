#!/usr/bin/bash

# this job script studies the relationship between batch size as determined as a quanitity decreasing (per device batch size) with number of gpu_devices available
# to train one epoch of the validation set of enamine (~40m molecules) 


sbatch --nodes 1 --job-name enamine_1_128_strong enamine_strong_scaling_test_full_set.sl 256
sbatch --nodes 2 --job-name enamine_2_128_strong enamine_strong_scaling_test_full_set.sl 256
sbatch --nodes 4 --job-name enamine_4_128_strong enamine_strong_scaling_test_full_set.sl 256
sbatch --nodes 8 --job-name enamine_8_128_strong enamine_strong_scaling_test_full_set.sl 256
sbatch --nodes 16 --job-name enamine_16_128_strong enamine_strong_scaling_test_full_set.sl 256
sbatch --nodes 32 --job-name enamine_32_128_strong enamine_strong_scaling_test_full_set.sl 256

