# LBANN: Livermore Big Artificial Neural Network Toolkit

## Building LBANN with Spack [for Users]

   spack install lbann

## Building LBANN with Spack [for Developers]

### Using spack setup

    http://spack.readthedocs.io/en/latest/packaging_guide.html?highlight=spack%20diy#build-system-configuration-support

    cd lbann
    spack setup lbann@local %intel@18.0.0 ^mvapich2
    mkdir spack-build; cd spack-build
    ../spconfig.py ..
    make
    make install

### Using spack diy
   spack diy lbann@local %gcc@4.9.3
   spack diy lbann@local %intel@17.0.0
 
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
