# LBANN: Livermore Big Artificial Neural Network Toolkit

## Building LBANN with Spack [for Users]

   spack install lbann

## Building LBANN with Spack [for Developers]

### Installing a compiler (if needed)

LBANN uses C++ features provided by newer compilers.  If you do not have the necessary compiler, you can use spack to install one.  For full details, see the [spack documentation](http://spack.readthedocs.io/en/latest/getting_started.html#compiler-configuration).

    spack install gcc@7.1.0

    The above command builds and installs a compiler.  It prints the install path as the final line.  If successful, then register this compiler with spack using the spack compiler find command, passing the install path as an argument.

        spack compiler add /path/to/compiler/install

### Using spack setup

Here is an example of setting up the local build environment on x86\_64 HPC system

    cd lbann
        mkdir spack_builds; cd spack_builds
            ../scripts/spack_receipes/build_lbann.sh -c gcc@7.1.0 -b openblas -m mvapich2
                cd gcc-7.1.0_x86_64_mvapich2_openblas_rel/build
                    make -j all

                    [Spack Setup](http://spack.readthedocs.io/en/latest/packaging_guide.html?highlight=spack%20diy#build-system-configuration-support)

                    The build\_lbann.sh script roughly does the following steps for this example:

                        spack setup lbann@local build_type=Release dtype=4 %gcc@7.1.0 ^elemental@master blas=openblas ^mvapich2
                            spack setup lbann@local %intel@18.0.0 ^mvapich2
                                mkdir -p gcc-7.1.0_x86_64_mvapich2_openblas_rel/build
                                    cd gcc-7.1.0_x86_64_mvapich2_openblas_rel/build
                                        ../spconfig.py ../../..

                                        By default, MVAPICH2 builds for PSM.  For an ibverbs build of MVAPICH2, use the following:

                                            ../scripts/spack_receipes/build_lbann.sh -c gcc@7.1.0 -b openblas -m 'mvapich2 fabrics=mrail'

## Using LBANN on LLNL LC clusters

    cd examples
     
## Setup the environment

         source setup_brain_lbann_env.sh

## Compile:

NOTE: Compilation is now done using cmake, please follow the instruction in doc/getting\_started 

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

