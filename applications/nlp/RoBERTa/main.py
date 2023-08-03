from types import SimpleNamespace
import argparse
import os
import sys
import json
import numpy as np

import lbann
import lbann.contrib.args
import lbann.contrib.launcher

from lbann.models import RoBERTa

# ----------------------------------------------
# Options
# ----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="number of epochs to train",
)
parser.add_argument(
    "--mini-batch-size",
    default=32,
    type=int,
    help="size of minibatches for training",
)
parser.add_argument(
    "--job-name",
    action="store",
    default="lbann_RoBERTa",
    type=str,
    help="scheduler job name",
    metavar="NAME",
)
parser.add_argument(
    "--work-dir",
    action="store",
    default=None,
    type=str,
    help="working directory",
    metavar="DIR",
)
parser.add_argument("--batch-job", action="store_true", help="submit as batch job")
parser.add_argument(
    "--checkpoint", action="store_true", help="checkpoint trainer after every epoch"
)
lbann.contrib.args.add_scheduler_arguments(parser)
lbann_params = parser.parse_args()

# ----------------------------------------------
# Data Reader
# ----------------------------------------------
def make_data_reader():
    reader = lbann.reader_pb2.DataReader()

    # Train data reader
    _reader = reader.reader.add()
    _reader.name = "python"
    _reader.role = "train"
    _reader.shuffle = True
    _reader.fraction_of_data_to_use = 1.0
    _reader.python.module = "dataset"
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = "get_sample"
    _reader.python.num_samples_function = "num_samples"
    _reader.python.sample_dims_function = "sample_dims"

    # Validation data reader
    _reader = reader.reader.add()
    _reader.name = "python"
    _reader.role = "validate"
    _reader.shuffle = False
    _reader.fraction_of_data_to_use = 1.0
    _reader.python.module = "dataset"
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = "get_sample"
    _reader.python.num_samples_function = "num_samples"
    _reader.python.sample_dims_function = "sample_dims"

    # Test data reader
    _reader = reader.reader.add()
    _reader.name = "python"
    _reader.role = "test"
    _reader.shuffle = False
    _reader.fraction_of_data_to_use = 1.0
    _reader.python.module = "dataset"
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = "get_sample"
    _reader.python.num_samples_function = "num_samples"
    _reader.python.sample_dims_function = "sample_dims"

    return reader


# ----------------------------------------------
# Loss
# ----------------------------------------------
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
        data_layout="data_parallel",
    ):
        self.num_classes = num_classes
        self.data_layout = data_layout
        self.fc = lbann.modules.FullyConnectedModule(
            self.num_classes,
            weights=weights,
            bias=False,
            activation=lbann.LogSoftmax,
            name="class_fc",
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
            mode="sum",
            data_layout=self.data_layout,
        )
        loss = lbann.Negative(loss, data_layout=self.data_layout)
        return loss


# ----------------------------------------------
# Build and Run Model
# ----------------------------------------------
with open("./config.json") as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
config.input_shape = (16, 32)
config.load_weights = os.path.abspath('./pretrained_weights')

# Construct the model
input_ = lbann.Slice(
    lbann.Input(data_field="samples"),
    slice_points=[0, 1, 1 + np.prod(config.input_shape)],
)
labels = lbann.Identity(input_)
sample = lbann.Reshape(input_, dims=config.input_shape)
roberta = RoBERTa(config, load_weights=config.load_weights)
out = roberta(sample)
out = lbann.ChannelwiseFullyConnected(out, output_channel_dims=[1000])
loss = CrossEntropyLoss(10, data_layout="model_parallel")
obj = loss(out, labels)
metrics = [lbann.Metric(obj, name="loss")]

model = lbann.Model(
    lbann_params.epochs,
    layers=lbann.traverse_layer_graph(input_),
    objective_function=obj,
    metrics=metrics,
    callbacks=[
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
    ],
)

# Setup trainer, optimizer, data_reader
trainer = lbann.Trainer(
    mini_batch_size=lbann_params.mini_batch_size,
    num_parallel_readers=1,
)
optimizer = lbann.Adam(
    learn_rate=0.01,
    beta1=0.9,
    beta2=0.99,
    eps=1e-8,
)
data_reader = make_data_reader()

# Launch LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(lbann_params)
kwargs["environment"] = {}
lbann.contrib.launcher.run(
    trainer,
    model,
    data_reader,
    optimizer,
    work_dir=lbann_params.work_dir,
    job_name=lbann_params.job_name,
    lbann_args=["--num_io_threads=1"],
    batch_job=lbann_params.batch_job,
    **kwargs,
)
