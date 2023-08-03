import gzip
import os
import os.path
import urllib.request

import google.protobuf.text_format
import lbann

# Paths
data_dir = os.path.dirname(os.path.realpath(__file__))

def download_data():
    """Download MNIST data files, if needed.

    Data files are downloaded from http://yann.lecun.com/exdb/mnist/
    and uncompressed. Does nothing if the files already exist.

    """

    # MNIST data files and associated URLs
    urls = {
        'train-images-idx3-ubyte': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
    }

    # Download and uncompress MNIST data files, if needed
    for data_file, url in urls.items():
        data_file = os.path.join(data_dir, data_file)
        compressed_file = data_file + '.gz'
        if not os.path.isfile(data_file):
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'LBANN/vision-app'},
            )
            with urllib.request.urlopen(request) as response, \
                 open(compressed_file, 'wb') as out_file:
                out_file.write(response.read())
            with gzip.open(compressed_file, 'rb') as in_file, \
                 open(data_file, 'wb') as out_file:
                out_file.write(in_file.read())

def make_data_reader(validation_fraction=0.1):
    """Make Protobuf message for MNIST data reader.

    MNIST data is downloaded if needed.

    Args:
        validation_fraction (float): The proportion of samples to be tested
        as the validation dataset.

    """

    # Download MNIST data files
    download_data()

    # Load Protobuf message from file
    protobuf_file = os.path.join(data_dir, 'data_reader.prototext')
    message = lbann.lbann_pb2.LbannPB()
    with open(protobuf_file, 'r') as f:
        google.protobuf.text_format.Merge(f.read(), message)
    message = message.data_reader

    if validation_fraction is not None:
        assert message.reader[0].role == "train"
        message.reader[0].validation_fraction = validation_fraction

    # Set paths
    for reader in message.reader:
        reader.data_filedir = data_dir

    return message
