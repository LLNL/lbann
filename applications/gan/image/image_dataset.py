import numpy as np
import os
from glob import glob
from PIL import Image, ImageOps
import lbann

class ImageDataset:
    def __init__(self, data_path):
        self.files = glob(os.path.abspath(data_path))
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = ImageOps.fit(Image.open(self.files[idx]), size=(128, 128)).convert('RGB')
        return np.array(img).astype(np.float32).transpose([2, 0, 1]).ravel() / 255

data_path = os.environ['GAN_DATA_PATH']

md = ImageDataset(data_path)

def get_sample(i):
    return md[i]

def num_samples():
    return len(md)

def sample_dims():
    return (3 * 128 * 128,)

def make_data_reader(fraction_of_data_to_use=1):
    reader = lbann.reader_pb2.DataReader()

    _reader = reader.reader.add()
    _reader.name = "python"
    _reader.role = "train"
    _reader.shuffle = True
    _reader.fraction_of_data_to_use = fraction_of_data_to_use
    _reader.python.module = "image_dataset"
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = "get_sample"
    _reader.python.num_samples_function = "num_samples"
    _reader.python.sample_dims_function = "sample_dims"

    return reader