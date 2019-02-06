# LBANN: Livermore Big Artificial Neural Network Toolkit

The Livermore Big Artificial Neural Network toolkit (LBANN) is an
open-source, HPC-centric, deep learning training software stack that
is optimized to compose multiple levels of parallelism.  LBANN is
built upon a distributed linear algebra library
(https://github.com/LLNL/Elemental) and provides model parallel
acceleration through domain decomposition to optimize for strong
scaling of network training.  LBANN is also able to compose model
parallelism with both data parallelism and ensemble training methods
for training large neural networks with massive amounts of data.
LBANN is a C++ code that composes MPI+OpenMP with CUDA (plus cuDNN and
cuBLAS), taking advantage of tightly coupled accelerators, low-latency
high bandwidth networking, node-local NVRAM storage, and high
bandwidth parallel file systems.  We have also developed an open
source asynchronous all-reduce library called Aluminum
(https://github.com/LLNL/Aluminum), that provides efficient MPI
implementations of communication algorithms that are optimized for
deep learning training patterns, provide GPU-accelerated reduction
kernels, and compose with OpenMP threaded code bases.  Aluminum also
includes support for using NVIDIA’s NCCL library and has been
integrated into both LBANN and Hydrogen (our distributed dense linear
algebra library).

LBANN supports state of the art training algorithms such as
unsupervised, self-supervised, and generative (GAN) training methods
in addition to traditional supervised learning.  It also supports
recurrent neural networks via back propagation through time (BPTT)
training, transfer learning, and multi-model training methods such as
the Livermore Tournament Fast Batch (LTFB) ensemble algorithm.


## Building LBANN
The LBANN build system is documented [here](docs/BuildingLBANN.md#top).

## LBANN Container Builds

We provide basic container defintion files, and instructions for their
use, in the containers subdirectory. We currently support Docker and
Singularity.

## Cmake (Non LC or OSX Systems/Script alternative)
   1. Ensure the following dependencies are installed
       * [CMake](https://software.llnl.gov/lbann/cmake.html)
       * [MPI](https://software.llnl.gov/lbann/mpi.html)
       * [Elemental](https://software.llnl.gov/lbann/elemental.html)
       * [OpenCV](https://software.llnl.gov/lbann/opencv.html)
       * CUDA (optional)
       * cuDNN (optional)
       * [Protocol Buffers](https://software.llnl.gov/lbann/protobuf.html) (optional)
       * [Doxygen](https://software.llnl.gov/lbann/doxygen.html) (optional)
       * *Note: LBANN also requires a C++ compiler with OpenMP support. The GCC 5.0 and Intel 16.0 C++ compilers are recommended*
    2. Clone this repo using `git clone https://github.com/LLNL/lbann.git`
    3. In the main LBANN directory create a build directory using `mkdir build`
    4. `cd` into this directory and run the following commands
       ```shell
       cmake ../..
       make
       make install
       ```
       * *Note: It may be necessary to manually set CMake variables to control the build configuration*

## Verifying LBANN on LC
   1. Allocate compute resources using SLURM: `salloc -N1 -t 60`
   2. Run a test experiment for the MNIST data set; from the main lbann directory run the following command:
 ```shell
  srun -n12 build/gnu.catalyst.llnl.gov/install/bin/lbann \
--model=model_zoo/models/lenet_mnist/model_lenet_mnist.prototext \
--reader=model_zoo/data_readers/data_reader_mnist.prototext \
--optimizer=model_zoo/optimizers/opt_adagrad.prototext \
--num_epochs=5
```
Note: `srun -n12 build/gnu.catalyst.llnl.gov/install/bin/lbann` assumes you are running on the LLNL catalyst platform;
  if running on some other platform, and/or have installed lbann in a different directory, you
  will need to adjust this command.

  This should produce roughly the following final results on Catalyst:
```
--------------------------------------------------------------------------------
[4] Epoch : stats formated [tr/v/te] iter/epoch = [844/94/157]
            global MB = [  64/  64/  64] global last MB = [  48  /  48  /  16  ]
             local MB = [  64/  64/  64]  local last MB = [  48+0/  48+0/  16+0]
--------------------------------------------------------------------------------
Model 0 training epoch 4 objective function : 0.0471567
Model 0 training epoch 4 categorical accuracy : 99.6241%
Model 0 training epoch 4 run time : 7.64182s
Model 0 training epoch 4 mini-batch time statistics : 0.00901458s mean, 0.0212693s max, 0.0078979s min, 0.000458463s stdev
Model 0 validation objective function : 0.0670221
Model 0 validation categorical accuracy : 98.9%
Model 0 validation run time : 0.25341s
Model 0 validation mini-batch time statistics : 0.00269454s mean, 0.00285273s max, 0.0020936s min, 6.65695e-05s stdev
Model 0 test objective function : 0.0600125
Model 0 test categorical accuracy : 99.02%
Model 0 test run time : 0.421912s
Model 0 test mini-batch time statistics : 0.00268631s mean, 0.00278771s max, 0.00131827s min, 0.00011085s stdev
```
  Note: LBANN performance will vary on a machine to machine basis. Results will also vary, but should not do so significantly.

## Running other models
There are various prototext models under the `lbann/model_zoo/models/` directory:
* `alexnet`
* `autoencoder_mnist`
* `lenet_mnist`

and etc.

Each of these directories should have a script called `runme.py`.
Run this script with no command line parameters for complete usage.
Basically, these scripts generate command lines similar to the one above
(in the *Verifying LBANN on LC* section).
The scripts take two required arguments: `--nodes=<int> and --tasks=<int>`.
The `tasks` option is used to specify the number of tasks per node, hence,
the total number of tasks (cores) is: `nodes`\*`tasks`.
The generated command lines are designed to be executed using `srun`
on LC systems, so you may need to modify, e.g, substitute mpirun,
depending on your specific system.

Note: some directories contain multiple models, e.g, as of this writing,
the `autoencoder_cifar10` directory contains both `model_autoencoder_cifar10.prototext`
and `model_conv_autoencoder_cifar10.prototext`.
In these cases there may be multiple python scripts, e.g, `runme_conv.py`.
