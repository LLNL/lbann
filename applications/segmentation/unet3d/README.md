## 3D The U-Net
LBANN implementation of the 3D U-Net:

> Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, and Olaf Ronneberger. "3D U-Net: learning dense volumetric segmentation from sparse annotation." In International conference on medical image computing and computer-assisted intervention, pp. 424-432, 2016.

This model requires Distconv.

### How to prepare a dataset for the 3D U-Net
By default, the HDF5 data reader expects a directory that contains `.hdf5` files.
Each HDF5 file represents a single data sample, and contains two different HDF5 datasets:
* `volume` that is a 3D array in `LBANN_DATATYPE` (such as `float`)
* `segmentation` that is a 3D array with the same shape and the data type as `volume`

Run `python3 applications/utils/verify_hdf5_dataset.py /path/to/hdf5/files --key-data volume --key-labels segmentation` to check whether the dataset format is appropriate for LBANN.

### How to Train
Run `python3 ./unet3d.py`.
See `python3 ./unet3d.py --help` for more options.
