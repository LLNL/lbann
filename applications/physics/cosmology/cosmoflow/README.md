## The CosmoFlow Network
LBANN implementation of the CosmoFlow network:

> Amrita Mathuriya, Deborah Bard, Peter Mendygral, Lawrence Meadows, James Arnemann, Lei Shao, Siyu He, Tuomas Karna, Diana Moise, Simon J. Pennycook, Kristyn Maschhoff, Jason Sewall, Nalini Kumar, Shirley Ho, Michael F. Ringenburg, Prabhat, and Victor Lee. "Cosmoflow: Using deep learning to learn the universe at scale." Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis, SC'18, pp. 65:1-65:11, 2018.

This model requires Distconv.

### How to prepare a dataset for the CosmoFlow network
By default, the HDF5 data reader expects a directory that contains `.hdf5` files.
Each HDF5 file represents a single data sample, and contains two different HDF5 datasets:
* `full` that is a 3D array in `LBANN_DATATYPE` (such as `float`)
* `unitPar` that is a 1D array with the same shape and the data type as `volume`

Run `python3 applications/utils/verify_hdf5_dataset.py /path/to/hdf5/files --key-data full --key-labels unitPar` to check whether the dataset format is appropriate for LBANN.

### How to Train
Run `python3 ./cosmoflow.py`.
See `python3 ./cosmoflow.py --help` for more options.
