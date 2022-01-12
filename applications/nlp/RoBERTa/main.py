from collections import namedtuple
import argparse
import os
import sys
import json
import nltk

import lbann
from lbann.util import str_list
import lbann.contrib.args
import lbann.contrib.launcher

from roberta import RobertaModel

root_dir = os.path.dirname(os.path.realpath(__file__))

# ----------------------------------------------
# Options
# ----------------------------------------------

# Command-line options
parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-name', action='store', default='lbann_yubnub', type=str,
    help='scheduler job name', metavar='NAME')
parser.add_argument(
    '--work-dir', action='store', default=None, type=str,
    help='working directory', metavar='DIR')
parser.add_argument(
    '--batch-job', action='store_true',
    help='submit as batch job')
parser.add_argument(
    '--dump-embeddings', action='store_true',
    help='dump author embeddings from validation set to .npy files')
parser.add_argument(
    '--checkpoint', action='store_true',
    help='checkpoint trainer after every epoch')
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    'yubnub_args', nargs=argparse.REMAINDER,
    help='options to pass into yubnub (separate from LBANN options with "--")',
)
lbann_params = parser.parse_args()

def make_data_reader():
    reader = lbann.reader_pb2.DataReader()

    # Train data reader
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = 'dataset'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_sample'
    _reader.python.num_samples_function = 'num_samples'
    _reader.python.sample_dims_function = 'sample_dims'

    # Validation data reader
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'validate'
    _reader.shuffle = False
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = 'dataset'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_sample'
    _reader.python.num_samples_function = 'num_samples'
    _reader.python.sample_dims_function = 'sample_dims'

    # Test data reader
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'test'
    _reader.shuffle = False
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = 'dataset'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_sample'
    _reader.python.num_samples_function = 'num_samples'
    _reader.python.sample_dims_function = 'sample_dims'

    return reader

class CrossEntropyLoss(lbann.modules.Module):
    """Cross-entropy loss for classification.

    Given an input vector x, weight matrix W, and label y:

      L = -log( softmax(W*x) * onehot(y) )

    Args:
      num_classes (int): Number of class.
      weights (lbann.Weights): Matrix with dimensions of
        num_classes x input_size. Each row is an embedding vector
        for the corresponding class.
      data_layout (str): Data layout of fully-connected layer.

    """
    def __init__(
            self,
            num_classes,
            weights=[],
            data_layout='data_parallel',
    ):
        self.num_classes = num_classes
        self.data_layout = data_layout
        self.fc = lbann.modules.FullyConnectedModule(
            self.num_classes,
            weights=weights,
            bias=False,
            activation=lbann.LogSoftmax,
            name='class_fc',
            data_layout=self.data_layout,
        )

    def forward(self, x, label):
        """Compute cross-entropy loss.

        Args:
          x (lbann.Layer): Input vector.
          label (lbann.Layer): Label. Should have one entry, which
            will be cast to an integer.

        Returns:
          lbann.Layer: Loss function value.

        """
        log_probs = self.fc(x)
        label_onehot = lbann.OneHot(
            label,
            size=self.num_classes,
            data_layout=self.data_layout,
        )
        loss = lbann.Multiply(
            log_probs,
            label_onehot,
            data_layout=self.data_layout,
        )
        loss = lbann.Reduction(
            loss,
            mode='sum',
            data_layout=self.data_layout,
        )
        loss = lbann.Negative(loss, data_layout=self.data_layout)
        return loss 

# ----------------------------------------------
# Run LBANN
# ----------------------------------------------

def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

with open('./config.json') as f:
    config = json.load(f)
config['is_decoder'] = False
config['add_cross_attention'] = False
config['output_attentions'] = False
config['use_return_dict'] = True
config['batch_size'] = 32
config['input_shape'] = (16,32)
config = convert(config)

# Construct the model
input_ = lbann.Slice(
    lbann.Input(data_field="samples"), slice_points=str_list([0, 1, 1 + 16 * 32])
)
labels = lbann.Identity(input_)
sample = lbann.Reshape(input_, dims=str_list([16, 32]))
roberta = RobertaModel(config, load_weights=True)
out = roberta(sample)
out = lbann.ChannelwiseFullyConnected(out, output_channel_dims=[1000])
loss = CrossEntropyLoss(10, data_layout="model_parallel")
obj = loss(out, labels)
metrics = [lbann.Metric(obj, name="loss")]

model = lbann.Model(
    0,
    layers=lbann.traverse_layer_graph(input_),
    objective_function=obj,
    metrics=metrics,
    callbacks=[
	lbann.CallbackPrintModelDescription(),
	lbann.CallbackPrint(),
	lbann.CallbackTimer(),
	lbann.CallbackDumpOutputs(),
    ],
)

# Setup trainer, optimizer, data_reader
trainer = lbann.Trainer(
    mini_batch_size=4,
    num_parallel_readers=1,
)
optimizer = lbann.Adam(
    learn_rate=0,
    beta1=0.9,
    beta2=0.99,
    eps=1e-8,
)
data_reader = make_data_reader()

# Run LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(lbann_params)
kwargs['environment'] = {}
lbann.contrib.launcher.run(
    trainer,
    model,
    data_reader,
    optimizer,
    work_dir=lbann_params.work_dir,
    job_name=lbann_params.job_name,
    lbann_args=['--num_io_threads=1'],
    batch_job=lbann_params.batch_job,
    **kwargs,
)
