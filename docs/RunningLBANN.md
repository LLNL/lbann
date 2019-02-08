# Running LBANN

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

A list of options for LBANN may be found by running `lbann
--help`. **NOTE**: At time of writing, it is known that some of these
are out-of-date. An [issue](https://github.com/LLNL/lbann/issues/864)
has been opened to track this.

## Using the model zoo

LBANN ships with prototext descriptions of a variety of models,
optimizers and data readers. These may be found in the `model_zoo/`
directory of the source repository or the `share/model_zoo/` directory
of the install directory.

**WARNING**: Some of these prototexts point to specific data locations
on LLNL LC clusters. Users may have to modify such paths to point to
locations on their own systems. This can be done by modifying the
prototext directly or overriding the options on the command line with,
e.g., the `--data_filedir_train` and `--data_filedir_test` options.

The following is an example invocation of LBANN on a machine using
Slurm's `srun` as an MPI launcher. In the example command,
a machine with 2 GPGPUs per node are available, 4 nodes will be used,
`${LBANN_EXE}` is the path to the `lbann` executable, and
`${LBANN_MODEL_ZOO_DIR}` is the path to the `model_zoo/` directory in
either the source tree or the install tree. Note that the options
passed to `srun` are not likely to be portable to other MPI
launchers. The example will train Alexnet with SGD optimization on the
Imagenet dataset for 5 epochs.
```bash
srun -N4 --ntasks-per-node=2 \
    ${LBANN_EXE} \
    --model=${LBANN_MODEL_ZOO_DIR}/models/alexnet/alexnet.prototext \
    --optimizer=${LBANN_MODEL_ZOO_DIR}/optimizers/opt_sgd.prototext \
    --reader=${LBANN_MODEL_ZOO_DIR}/data_readers/data_reader_imagenet.prototext
    --num_epochs=5
```

## Using the Python interface for prototext

There is a python interface for generating model prototext
files. Example Python scripts may be found in the
`scripts/proto/lbann/models` directory of the source
repository. Running the Python script will generate a prototext that
can be passed to the `--model` option for LBANN.

```bash
python3 alexnet.py alexnet.prototext
<mpi-launcher> <mpi-options> \
    lbann --model=alexnet.prototext <other-lbann-options>
```

where `<other-lbann-options>` are as documented <a
href="#using-the-model-zoo">above</a>, with optimizer and data reader
prototexts coming from the appropriate `model_zoo/` directories.

## Running the inference engine

This section is under construction, requiring input from other team
members. Until it is complete, please ask questions on the [issue
tracker](https://github.com/llnl/lbann/issues).
