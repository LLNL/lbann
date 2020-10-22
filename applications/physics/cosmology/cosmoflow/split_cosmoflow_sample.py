#!/usr/bin/env python3

import argparse
import os
import re
import sys
from itertools import product

import h5py
import numpy as np

if __name__ == "__main__":
    ORIG_WIDTH = 512
    ORIG_NUM_PARAMS = 4

    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_files", nargs="*",
                        help="Path to a CosmoFlow HDF5 file.")
    parser.add_argument("--out_dir", type=str, default="dataset",
                        help="An  optional value.")
    parser.add_argument("--width", type=int, default=128,
                        help="The output spatial width.")
    parser.add_argument("--datatype", type=str, default="float32",
                        help="The data type for universe data.")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir) or not os.path.isdir(args.out_dir):
        sys.stderr.write("The output directory does not exist: {}\n"
                         .format(args.out_dir))
        exit(1)

    if (ORIG_WIDTH % args.width) != 0:
        sys.stderr.write("The output width is not a divisor of the original width({}): {}\n"
                         .format(ORIG_WIDTH, args.width))
        exit(1)

    if args.datatype not in ["float", "float32", "int16"]:
        sys.stderr.write("Unrecognized data type: {}\n".format(args.datatype))

    data_type = getattr(np, args.datatype)
    sub_cube_count = ORIG_WIDTH // args.width
    for hdf5_file in args.hdf5_files:
        m = re.compile("(.*)\\.hdf5$").match(os.path.basename(hdf5_file))
        if m is None:
            sys.stderr.write("Unrecognized file name: {}\n".format(hdf5_file))
            exit(1)

        hdf5_file_wo_ext = m.group(1)

        h = h5py.File(hdf5_file, "r")
        full = h["full"]
        unitPar = h["unitPar"]
        assert full.value.shape == tuple([ORIG_WIDTH]*3+[ORIG_NUM_PARAMS])
        assert unitPar.value.shape == (ORIG_NUM_PARAMS,)
        full_transposed = full.value.transpose().astype(data_type)

        for ix, iy, iz in product(range(sub_cube_count),
                                  range(sub_cube_count),
                                  range(sub_cube_count)):
            cube = full_transposed[
                :,
                (args.width*ix):(args.width*(ix+1)),
                (args.width*iy):(args.width*(iy+1)),
                (args.width*iz):(args.width*(iz+1)),
            ]
            assert cube.shape == tuple([ORIG_NUM_PARAMS]+[args.width]*3)

            out_path = os.path.join(
                args.out_dir,
                "{}_{}_{}_{}.hdf5".format(hdf5_file_wo_ext, ix, iy, iz))

            with h5py.File(out_path, "w-") as hw:
                hw["full"] = cube
                hw["unitPar"] = unitPar.value
