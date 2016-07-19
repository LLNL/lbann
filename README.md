# LBANN: Livermore Big Artificial Neural Network Toolkit
 
## Using LBANN on LLNL LC clusters

    cd examples
 
## Setup the environment

    source setup_brain_lbann_env.sh

## Compile:

NOTE: Compilation is now done using cmake, please follow the instruction in doc/getting_started 

    make

## Running on Catalyst

### Interactive Mode 

Allocate nodes in SLURM:

    salloc -N16 --enable-hyperthreads -t 1440 --clear-ssd
    ./run_lbann_dnn_imagenet.sh -t 2400 -v 10 -e 4 -n 5000,2500,1000 -b 192 -r 0.0001

### Batch Mode

    cd tests
    sbatch -N16 --enable-hyperthreads -t 1440 --clear-ssd ./test_imagenet_topologies.sh

## Running on Surface

### Interactive Mode

Allocate nodes in MOAB:

    mxterm 16 256 1440 -A hpclearn
