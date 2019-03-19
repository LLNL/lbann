# LBANN: Livermore Big Artificial Neural Network Toolkit

The Livermore Big Artificial Neural Network toolkit (LBANN) is an
open-source, HPC-centric, deep learning training framework that is
optimized to compose multiple levels of parallelism.

LBANN provides model-parallel acceleration through domain
decomposition to optimize for strong scaling of network training.  It
also allows for composition of model-parallelism with both data
parallelism and ensemble training methods for training large neural
networks with massive amounts of data.  LBANN is able to advantage of
tightly-coupled accelerators, low-latency high-bandwidth networking,
and high-bandwidth parallel file systems.

LBANN supports state-of-the-art training algorithms such as
unsupervised, self-supervised, and adversarial (GAN) training methods
in addition to traditional supervised learning.  It also supports
recurrent neural networks via back propagation through time (BPTT)
training, transfer learning, and multi-model and ensemble training
methods.


## Building LBANN
The preferred method for LBANN users to install LBANN is to use
[Spack](https://github.com/llnl/spack). After some system
configuration, this should be as straightforward as

```bash
spack install lbann
```

More detailed instructions for building and installing LBANN are
available at the [main LBANN
documentation](https://lbann.readthedocs.io/en/latest/index.html).

## Running LBANN
The basic template for running LBANN is

```bash
<mpi-launcher> <mpi-options> \
    lbann <lbann-options> \
    --model=model.prototext \
    --optimizer=opt.prototext \
    --reader=data_reader.prototext
```

When using GPGPU accelerators, users should be aware that LBANN is
optimized for the case in which one assigns one GPU per MPI
*rank*. This should be borne in mind when choosing the parameters for
the MPI launcher.

More details about running LBANN are documented
[here](https://lbann.readthedocs.io/en/latest/running_lbann.html).

## Publications

A list of publications, presentations and posters are shown
[here](https://lbann.readthedocs.io/en/latest/publications.html).

## Reporting issues
Issues, questions, and bugs can be raised on the [Github issue
tracker](https://github.com/llnl/lbann/issues).
