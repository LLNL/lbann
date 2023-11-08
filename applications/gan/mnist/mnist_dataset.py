import numpy as np
import os
import gzip
import os.path
import urllib.request
import lbann

data_dir = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(data_dir, 'images')

# Download training images if necessary.
if not os.path.exists(data_file):
    urllib.request.urlretrieve('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', data_file + '.gz')
    with gzip.open(data_file + '.gz', 'rb') as in_file, open(data_file, 'wb') as out_file:
        out_file.write(in_file.read())
    os.remove(data_file + '.gz')

class MnistDataset:
    def __init__(self):
        # Load and pad the images from 28x28 to 32x32.
        self.images = np.float32(np.fromfile(data_file, offset=16, dtype='uint8').byteswap().reshape(-1,28,28)) / 255
        self.images = np.pad(self.images, ((0,0), (2,2), (2,2))).reshape(-1, 32*32)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

md = MnistDataset()

def get_sample(i):
    return md[i]

def num_samples():
    return len(md)

def sample_dims():
    return (32*32,)

def make_data_reader(fraction_of_data_to_use=1):
    reader = lbann.reader_pb2.DataReader()

    _reader = reader.reader.add()
    _reader.name = "python"
    _reader.role = "train"
    _reader.shuffle = True
    _reader.fraction_of_data_to_use = fraction_of_data_to_use
    _reader.python.module = "mnist_dataset"
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = "get_sample"
    _reader.python.num_samples_function = "num_samples"
    _reader.python.sample_dims_function = "sample_dims"

    return reader