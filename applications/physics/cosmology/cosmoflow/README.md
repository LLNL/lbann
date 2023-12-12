# The CosmoFlow Network
LBANN implementation of the CosmoFlow network:

> Amrita Mathuriya, Deborah Bard, Peter Mendygral, Lawrence Meadows, James Arnemann, Lei Shao, Siyu He, Tuomas Karna, Diana Moise, Simon J. Pennycook, Kristyn Maschhoff, Jason Sewall, Nalini Kumar, Shirley Ho, Michael F. Ringenburg, Prabhat, and Victor Lee. "Cosmoflow: Using deep learning to learn the universe at scale." Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis, SC'18, pp. 65:1-65:11, 2018.

This model requires Distconv.

## How to prepare a dataset for the CosmoFlow network
1. Download the HDF5 version of the CosmoFlow dataset from [the official website](https://portal.nersc.gov/project/m3363/).
2. Run `python3 split_cosmoflow_sample.py --out_dir /path/to/output/dir --width WIDTH /path/to/cosmoflow/dataset/*.hdf5` to generate a dataset each sample of which has the spatial width of `WIDTH`.
3. Run `python3 applications/utils/verify_hdf5_dataset.py /path/to/output/dir --key-data full --key-responses unitPar` to check whether the dataset format is appropriate for LBANN.

### Datails about the dataset format
By default, the HDF5 data reader expects a directory that contains `.hdf5` files.
Each HDF5 file represents a single data sample, and contains two different HDF5 datasets:
* `full` that is a 4D array in `LBANN_DATATYPE` (such as `float`) format and the CDHW order (the first dimension is the channel dimension (C))
* `unitPar` that is a 1D array with the same data type as `volume`

## How to Train
Run `python3 ./train_cosmoflow.py`.
See `python3 ./train_cosmoflow.py --help` for more options.

## How to generate custom kernels for convolutions in DaCe

`cd DaCe_kernels`
`./generate.sh`
