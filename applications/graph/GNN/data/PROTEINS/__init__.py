import urllib.request
import tarfile 
import os
import os.path

import lbann 
 
data_dir = os.path.dirname(os.path.realpath(__file__))
def make_data_reader(data_format = "Sparse"):

    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True #Turn off shuffle for debugging 
    _reader.fraction_of_data_to_use = 1.0 
    _reader.python.module = f'PROTEINS_{data_format}_Dataloader'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_train'
    _reader.python.num_samples_function = 'num_train_samples' 
    _reader.python.sample_dims_function = 'sample_data_dims' 

    return reader 

if __name__ == '__main__':
    print(make_data_reader("Sparse"))
    print(make_data_reader("Dense"))