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

    parser.add_argument("--key-data", type=str,
                        help="The name of the \"data\" dataset.")
    parser.add_argument("--key-labels", type=str, default=None,
                        help="The name of the \"label\" dataset.")
    parser.add_argument("--key-responses", type=str, default=None,
                        help="The name of the \"response\" dataset.")

    args = parser.parse_args()
    print(args)

    if len(set([args.key_labels, args.key_responses]) - set([None])) != 1:
        sys.stderr.write("Error: Only either of label or response is required.\n")
        exit()

    hdf5_files = sorted([x for x in os.listdir(args.dataset_dir)
                         if re.compile(".*\\.hdf5$").match(x)])

    data_shape_set = set()
    label_shape_set = set()
    response_shape_set = set()
    data_dtype_set = set()
    label_dtype_set = set()
    response_dtype_set = set()
    for hdf5_file in hdf5_files:
        print(hdf5_file)
        fp = h5py.File(os.path.join(args.dataset_dir, hdf5_file), "r")
        line = []

        assert args.key_data in fp.keys()
        volume = fp[args.key_data].value
        line.append("data shape: {}, data type: {}, "
                    .format(volume.shape, volume.dtype))
        assert len(volume.shape) == 4
        data_shape_set.add(volume.shape)
        data_dtype_set.add(volume.dtype)

        if args.key_labels is not None:
            assert args.key_labels in fp.keys()
            labels = fp[args.key_labels].value
            line.append("label shape: {}, label type: {}"
                        .format(labels.shape, labels.dtype))
            assert len(labels.shape) == 4
            label_shape_set.add(labels.shape)
            label_dtype_set.add(labels.dtype)

        if args.key_responses is not None:
            assert args.key_responses in fp.keys()
            responses = fp[args.key_responses].value
            line.append("response shape: {}, response type: {}"
                        .format(responses.shape, responses.dtype))
            assert len(responses.shape) == 1
            response_shape_set.add(responses.shape)
            response_dtype_set.add(responses.dtype)

        print(" ".join(line))

    # The shape should be the same
    assert len(data_shape_set) == 1
    assert len(label_shape_set) <= 1
    assert len(response_shape_set) <= 1
    if args.key_labels is not None:
        assert data_shape_set == label_shape_set

    # Distconv expects each 3D tensor is a cube
    if args.key_labels is not None:
        spatial_dimensions = list(data_shape_set)[0][1:]
        assert len(spatial_dimensions) == 3
        if len(set(spatial_dimensions)) > 1:
            sys.stderr.write("Warning: The spatial dimensions are not the same."
                             " This might cause unexpected bugs."
                             " (found: {})\n".format(spatial_dimensions))

    # The data type should be the same
    assert len(data_dtype_set) == 1
    assert len(label_dtype_set) <= 1
    assert len(response_dtype_set) <= 1
    if args.key_labels is not None:
        assert data_dtype_set == label_dtype_set

    if args.key_responses is not None:
        assert data_dtype_set == response_dtype_set

    print("OK")
