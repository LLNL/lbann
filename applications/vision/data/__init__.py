import gzip
import os
import os.path
import urllib.request

import google.protobuf.text_format
import lbann

# Paths
data_dir = os.path.dirname(os.path.realpath(__file__))
mnist_dir = os.path.join(data_dir, 'mnist')

def download_mnist():
    """Download MNIST data files if needed.

    The files are uncompressed and saved in the 'mnist' directory.
    Does nothing if the files already exist.

    """

    # MNIST data files and associated URLs
    urls = {
        'train-images-idx3-ubyte': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    }

    # Download and uncompress MNIST data files, if needed
    for data_file, url in urls.items():
        data_file = os.path.join(mnist_dir, data_file)
        compressed_file = data_file + '.gz'
        if not os.path.isfile(data_file):
            urllib.request.urlretrieve(url, filename=compressed_file)
            with gzip.open(compressed_file, 'rb') as in_file:
                with open(data_file, 'wb') as out_file:
                    out_file.write(in_file.read())

def make_mnist_data_reader():
    """Make Protobuf message for MNIST data reader.

    MNIST data is downloaded if needed.

    """

    # Download MNIST data files
    download_mnist()

    # Load Protobuf message from file
    protobuf_file = os.path.join(mnist_dir, 'data_reader.prototext')
    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    # Set paths
    for reader in message.reader:
        reader.data_filedir = mnist_dir

    return message
