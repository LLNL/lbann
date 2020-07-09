import argparse
import os
import re
import sys

import h5py

if __name__ == '__main__':
    desc = ("Verify the content of a given HDF5 dataset directory."
            " Print \"OK\" if the format of the dataset is OK.")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("dataset_dir", help="Path to the HDF5 dataset.")
    args = parser.parse_args()
    print(args)

    hdf5_files = sorted([x for x in os.listdir(args.dataset_dir)
                         if re.compile(".*\\.hdf5$").match(x)])

    volume_shape_set = set()
    segmentation_shape_set = set()
    volume_dtype_set = set()
    segmentation_dtype_set = set()
    for hdf5_file in hdf5_files:
        print(hdf5_file)
        fp = h5py.File(os.path.join(args.dataset_dir, hdf5_file), "r")

        assert "volume" in fp.keys()
        assert "segmentation" in fp.keys()
        volume = fp["volume"].value
        segmentation = fp["segmentation"].value

        print("volume shape: {}, volume type: {}, "
              .format(volume.shape, volume.dtype) +
              "segmentation shape: {}, segmentation type: {}"
              .format(segmentation.shape, segmentation.dtype))

        assert len(volume.shape) == 4
        assert len(segmentation.shape) == 4

        volume_shape_set.add(volume.shape)
        segmentation_shape_set.add(segmentation.shape)
        volume_dtype_set.add(volume.dtype)
        segmentation_dtype_set.add(segmentation.dtype)

    # The shape should be the same
    assert len(volume_shape_set) == 1
    assert len(segmentation_shape_set) == 1
    assert volume_shape_set == segmentation_shape_set

    # Distconv expects each 3D tensor is a cube
    spatial_dimensions = list(volume_shape_set)[0][1:]
    assert len(spatial_dimensions) == 3
    if len(set(spatial_dimensions)) > 1:
        sys.stderr.write("Warning: The spatial dimensions are not the same."
                         " This might cause unexpected bugs."
                         " (found: {})\n".format(spatial_dimensions))

    # The data type should be the same
    assert len(volume_dtype_set) == 1
    assert len(segmentation_dtype_set) == 1
    assert volume_dtype_set == segmentation_dtype_set

    print("OK")
