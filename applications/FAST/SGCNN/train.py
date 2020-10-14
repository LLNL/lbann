import argparse
import lbann
import lbann.contrib.launcher
import lbann.contrib.args
from model import SGCNN
import os
import os.path as osp

desc = ("Training SGCNN on a Protein-ligand graphs using LBANN")
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--mini-batch-size",
                    action='store', default=32, type=int,
                    help="Mini batch size {default: 32}", metavar='NUM')
parser.add_argument("--num-epochs",
                    action='store', default=1, type=int,
                    help="Number of epochs {default: 1}", metavar='NUM')
parser.add_argument("--job-name",
                    action='store', default="SGCNN", type=str,
                    help="Job name {default: SGCNN}", metavar='NAME')
lbann.contrib.args.add_scheduler_arguments(parser)
args = parser.parse_args()
data_dir = os.path.dirname(os.path.realpath(__file__))


def make_data_reader(module_name='SimDataset',
                     module_dir=None,
                     sample_function='get_train',
                     num_samples_function='num_train_samples',
                     sample_dims='sample_dims'):
    module_dir = (module_dir if module_dir else
                  osp.dirname(osp.realpath(__file__)))
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = False  # Turn off shuffle for debugging
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = module_name
    _reader.python.module_dir = module_dir
    _reader.python.sample_function = sample_function
    _reader.python.num_samples_function = num_samples_function
    _reader.python.sample_dims_function = sample_dims

    return reader

def slice_data(input_layer):
    

def make_model():
    sgcnn_model = SGCNN()

    input_ = lbann.Input(target_mode = 'N/A')
