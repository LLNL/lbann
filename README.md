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

### Cmake (Non LC or OSX Systems/Script alternative)
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
   2. Run a test experiment from the main lbann directory using the following command:
   To Verify functionality of LBANN with a test MNIST experiment. Using the following command:
 ```
  srun -n2 build/catalyst.llnl.gov/model_zoo/lbann \
--model=model_zoo/historical/prototext/model_mnist_multi.prototext \
--reader=model_zoo/data_readers/data_reader_mnist.prototext \
--optimizer=model_zoo/optimizers/opt_adagrad.prototext
```
  This should produce the following final results on Catalyst:
  ```
  --------------------------------------------------------------------------------
  [20] Epoch : stats formated [tr/v/te] iter/epoch = [2700/600/1000]
  global MB = [  20/  10/  10] global last MB = [  20/  10/  10]
  local MB = [  10/  10/  10]  local last MB = [  10/  10/  10]
  --------------------------------------------------------------------------------
  Model 0 @54000 steps Training categorical accuracy: 99.9741% @12000 validation steps Validation categorical accuracy: 97.95%
  Model 1 @54000 steps Training categorical accuracy: 99.9926% @12000 validation steps Validation categorical accuracy: 97.9833%
  Model 0 average categorical cross entropy: 0.00176223
  Model 1 average categorical cross entropy: 0.00126326
  Model 0 Epoch time: 62.8982s; Mean minibatch time: 0.0218984s; Min: 0.0214709s; Max: 0.0306295s; Stdev: 0.000214515s
  Model 1 Epoch time: 62.9628s; Mean minibatch time: 0.0219376s; Min: 0.0216846s; Max: 0.127405s; Stdev: 0.00204183s
  Model 0 @20000 testing steps external validation categorical accuracy: 98.32%
  Model 1 @20000 testing steps external validation categorical accuracy: 98.27%
``` 
  LBANN performance will vary on a machine to machine basis. Results will also vary, but should not do so significantly. 

## Running other models
Launch an MPI job using the proper command for your system (srun, mpirun, mpiexec etc), calling the lbann executable found in lbann/build/$YourBuildSys/model_zoo. This executable requires three command line arguments. These arguments are prototext files specifying the model, optimizer and data reader for the execution. The files can be found in lbann/model_zoo/prototext. Models can be adjusted by altering these files. Example execution:
```
srun -n2 catalyst.llnl.gov/model_zoo/lbann
--model=../model_zoo/historical/prototext/model_mnist_multi.prototext
--reader=../model_zoo/data_readers/data_reader_mnist.prototext
--optimizer=../model_zoo/optimizers/opt_adagrad.prototext
```

