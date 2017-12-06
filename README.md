# LBANN: Livermore Big Artificial Neural Network Toolkit
## Building LBANN

### LC Systems
   1. Clone this repo using `git clone https://github.com/LLNL/lbann.git`
   2. From anywhere in the lbann directory run the LC build script located in  
   `~/lbann/scripts/build_lbann_lc.sh`
   3. This will build LBANN in a newly created build directory.

### OS X
   1. Clone this repo using `git clone https://github.com/LLNL/lbann.git`
   2. From anywhere in the lbann directory run the LC build script located in  
   `~/lbann/scripts/build_lbann_osx.sh`
   3. This will build LBANN in a newly created build directory.

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

## Running LBANN with Singularity

[Singularity](http://singularity.lbl.gov/)

Users can run LBANN inside a singularity container by grabbing the lbann.def found in the singularity directory, and running the following commands. 
```
singularity create -s 8000 lbann.img
sudo singularity bootstrap lbann.img lbann.def
```
*Note: Bootstrapping the image requires root access.*

This will create a container called lbann.img which can be used to invoke lbann on any system with singularity and openmpi installed.
To run LBANN use mpirun and singularity's execute command:
```
salloc -N2
mpirun -np 4 singularity exec -B /p:/p lbann.img /lbann/spack_builds/singularity_optimizied_test/model_zoo/lbann  mpirun -np 4 singularity exec -B /p:/p lbann.img /lbann/spack_builds/singularity_optimizied_test/model_zoo/lbann  --model=/lbann/model_zoo/tests/model_mnist_distributed_io.prototext --reader=/lbann/model_zoo/data_readers/data_reader_mnist.prototext --optimizer=/lbann/model_zoo/optimizers/opt_adagrad.prototext 
```
Note: User must include the -B singularity command, to bind any necessary files to the container. This includes user generated prototext files, and any datasets needed. Alternatively, system admins are capable of allowing a singularity container to utilize the host's filesystem. This is done by changing the MOUNT HOSTFS in the singularity config file.

## Cmake (Non LC or OSX Systems/Script alternative)
   1. Ensure the following dependencies are installed
    [CMake](https://software.llnl.gov/lbann/cmake.html)
    [MPI](https://software.llnl.gov/lbann/mpi.html)
    [Elemental](https://software.llnl.gov/lbann/elemental.html)
    [OpenCV](https://software.llnl.gov/lbann/opencv.html)
    CUDA (optional)
    cuDNN (optional)
    [Protocol Buffers](https://software.llnl.gov/lbann/protobuf.html) (optional)
    [Doxygen](https://software.llnl.gov/lbann/doxygen.html) (optional)
    *Note: LBANN also requires a C++ compiler with OpenMP support. The GCC 5.0 and Intel 16.0 C++ compilers are recommended*
    2. Clone this repo using `git clone https://github.com/LLNL/lbann.git`
    3. In the main LBANN directory create a build directory using `mkdir build`
    4. `cd` into this directory and run the following commands
    `cmake ../..`
    `make`
    `make install`
    *Note: It may be necessary to manually set CMake variables to control the build configuration*

## Verifying LBANN on LC
   1. Allocate compute resources using SLURM: `salloc -N1 -t 60`
   2. Run a test experiment for the MNIST data set; from the main lbann directory run the following command:
 ```
  srun -n12 build/catalyst.llnl.gov/model_zoo/lbann \
--model=model_zoo/tests/model_mnist_distributed_io.prototext \
--reader=model_zoo/data_readers/data_reader_mnist.prototext \
--optimizer=model_zoo/optimizers/opt_adagrad.prototext \
--num_epochs=5
```
Note: `srun -n12 build/catalyst.llnl.gov/model_zoo/lbann` assumes you are running on the LLNL catalyst platform;
  if running on some other platform, and/or have installed lbann in a different directory, you
  will need to adjust this command.

  This should produce the following final results on Catalyst:
```
--------------------------------------------------------------------------------
[5] Epoch : stats formated [tr/v/te] iter/epoch = [2700/600/1000]
            global MB = [  20/  10/  10] global last MB = [  20/  10/  10]
             local MB = [  10/  10/  10]  local last MB = [  10/  10/  10]
--------------------------------------------------------------------------------
Model 0 @13500 steps Training categorical accuracy: 99.3519% @3000 validation steps Validation categorical accuracy: 97.9167%
Model 1 @13500 steps Training categorical accuracy: 99.3222% @3000 validation steps Validation categorical accuracy: 97.6%
Model 0 average cross entropy: 0.079888
Model 1 average cross entropy: 0.0871397
Model 0 Epoch time: 30.1591s; Mean minibatch time: 0.010775s; Min: 0.0104503s; Max: 0.0126789s; Stdev: 0.000117069s
Model 1 Epoch time: 30.1591s; Mean minibatch time: 0.0107751s; Min: 0.010483s; Max: 0.0129022s; Stdev: 0.000118997s
Model 0 @1000 testing steps external validation categorical accuracy: 98.01%
Model 1 @1000 testing steps external validation categorical accuracy: 97.92%

``` 
  Note: LBANN performance will vary on a machine to machine basis. Results will also vary, but should not do so significantly. 

## Running other models
There are various prototext models under the lbann/model_zoo/models/ directory: alexnet, autoencoder_mnist, lenet_mnist, etc. Each of these directories should have a script called *runme.py*. Run this script with no command line parameters for complete usage. Basically, these scripts generate command lines similar to the one above (in the *Verifying LBANN on LC* section). The scripts take two required arguments: --nodes=`<int>` and --tasks=`<int>`. The "tasks" option is used to specify the number of tasks per node, hence, the total number of tasks (cores) is: nodes\*tasks. The generated command lines are designed to be executed using *srun* on LC systems, so you may need to modify, e.g, substitute mpirun, depending on your specific system.

Note: some directories contain multiple models, e.g, as of this writing, the autoencoder_cifar10 directory contains both *model_autoencoder_cifar10.prototext* and *model_conv_autoencoder_cifar10.prototext*. In these cases there may be multiple python scripts, e.g, *runme_conv.py*.

