#!/usr/bin/env python3

"""
Convert the MNIST training/test datasets into .npy and .npz files.

The generated files contain the following tensors:
* train.npy: shape=(60000, 785), dtype=np.float32
* test.npy:  shape=(10000, 785), dtype=np.float32
* train_int16.npz:
   * "data":   shape=(60000, 784), dtype=np.int16
   * "labels": shape=(60000, 1),   dtype=np.int32
* test_int16.npz:
   * "data":   shape=(10000, 784), dtype=np.int16
   * "labels": shape=(10000, 1),   dtype=np.int32

{train,test}.npy can be used for numpy_reader.
{train,test}_int16.npz can be used for numpy_npz_reader.
"""

import numpy as np
import argparse
import os

IMAGE_WIDTH = 28

def convert_mnist_to_np_and_npz(imagePath, labelPath,
                                imageMagicNumber, labelMagicNumber,
                                out, int16):
    with open(imagePath, "rb") as f:
        imageBin = f.read()

    assert imageMagicNumber == np.frombuffer(imageBin[ 0: 4], dtype=">u4")[0]
    imageCount              =  np.frombuffer(imageBin[ 4: 8], dtype=">u4")[0]
    assert IMAGE_WIDTH      == np.frombuffer(imageBin[ 8:12], dtype=">u4")[0]
    assert IMAGE_WIDTH      == np.frombuffer(imageBin[12:16], dtype=">u4")[0]
    pixels                  =  np.frombuffer(imageBin[16:], dtype=">u1") \
                                 .reshape([imageCount, IMAGE_WIDTH*IMAGE_WIDTH])

    with open(labelPath, "rb") as f:
        labelBin = f.read()

    assert labelMagicNumber == np.frombuffer(labelBin[ 0: 4], dtype=">u4")[0]
    assert imageCount       == np.frombuffer(labelBin[ 4: 8], dtype=">u4")[0]
    labels                  =  np.frombuffer(labelBin[8:], dtype=">u1") \
                                 .reshape([imageCount, 1])

    pixels = pixels.astype(np.float32) / 255.0
    labels = labels.astype(np.int32)

    npy = np.concatenate((pixels, labels.astype(np.float32)), axis=1)

    if int16:
        pixels = (pixels * 0x7FFF).astype(np.int16)

    np.save("{}.npy".format(out), npy)
    np.savez(
        "{}{}.npz".format(out, "_int16" if int16 else ""),
        data=pixels,
        labels=labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the MNIST training/test datasets into .npy and .npz files.",
        epilog="Usage: ./mnist_to_npy_and_npz.py path/to/mnist/directory")
    parser.add_argument(
        "mnist_dir", type=str,
        help="Path to a directory containing the MNIST dataset (decompressed binary files)")
    parser.add_argument(
        "--int16",
        dest="int16", action="store_const",
        const=True, default=True,
        help="Convert the image data into int16 (each pixel is multiplied by 0x7FFFF)")
    args = parser.parse_args()

    convert_mnist_to_np_and_npz(
        os.path.join(args.mnist_dir, "train-images-idx3-ubyte"),
        os.path.join(args.mnist_dir, "train-labels-idx1-ubyte"),
        2051, 2049,
        "train",
        args.int16)
    convert_mnist_to_np_and_npz(
        os.path.join(args.mnist_dir, "t10k-images-idx3-ubyte"),
        os.path.join(args.mnist_dir, "t10k-labels-idx1-ubyte"),
        2051, 2049,
        "test",
        args.int16)
