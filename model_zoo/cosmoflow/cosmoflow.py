#!/usr/bin/env python3
import argparse
import os.path
import google.protobuf.text_format as txtf
import lbann
import lbann.contrib.lc.launcher
import lbann.modules as lm
import lbann.proto as lp
from lbann.weights import Weights

import numpy as np

# ----------------------------------
# The CosmoFlow module
# ----------------------------------

class CosmoFlow(lm.Module):
    """
    CosmoFlow neural network.

    See:
        Amrita Mathuriya, Deborah Bard, Peter Mendygral, Lawrence Meadows,
        James Arnemann, Lei Shao, Siyu He, Tuomas Karna, Diana Moise,
        Simon J. Pennycook, Kristyn Maschhoff, Jason Sewall, Nalini Kumar,
        Shirley Ho, Michael F. Ringenburg, Prabhat, and Victor Lee.
        "Cosmoflow: Using deep learning to learn the universe at scale."
        Proceedings of the International Conference for High Performance
        Computing, Networking, Storage, and Analysis, SC'18, pp. 65:1-65:11,
        2018.

    Note that this model is somewhat different from the model.
    """

    global_count = 0  # Static counter, used for default names

    def __init__(self, output_size,
                 input_width,
                 name=None):
        """Initialize CosmFlow.

        Args:
            output_size (int): Size of output tensor.
            input_width (int): Width of input tensor.
            name (str, optional): Module name
                (default: 'cosmoflow_module<index>').

        """
        CosmoFlow.global_count += 1
        self.instance = 0
        self.name = (name if name
                     else 'cosmoflow_module{0}'.format(CosmoFlow.global_count))
        self.input_width = input_width
        assert self.input_width in [128, 256, 512]

        self.layer_params = [
            {"type": "conv", "out_channels": 16,  "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 32,  "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 64,  "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 128, "kernel_size": 3, "stride": 2},
            {"type": "pool"},
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "stride": 1},
            {"type": "pool"},
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "stride": 1},
            {"type": "conv", "out_channels": 256, "kernel_size": 3, "stride": 1},
        ]
        for p in self.layer_params:
            if p["type"] == "conv":
                p["padding"] = int((p["kernel_size"]-1)/2)

        additional_pools = []
        if self.input_width == 256:
            additional_pools = [6]
        elif self.input_width == 512:
            additional_pools = [6, 7]

        for i in additional_pools:
            conv_idx = list(np.cumsum([1 if x["type"] == "conv" else 0 for x in self.layer_params])).index(i)
            self.layer_params.insert(conv_idx+1, {"type": "pool"})

        width = self.input_width
        for p in self.layer_params:
            if p["type"] == "conv":
                output_width = int(width / p["stride"])
            else:
                output_width = int(width / 2)

            p["width"] = output_width
            width = output_width
            assert width > 0

        for i, param in enumerate(filter(lambda x: x["type"] == "conv", self.layer_params)):
            conv_name ="conv"+str(i+1)
            conv_weights = [Weights(initializer=lbann.GlorotUniformInitializer())]

            param_actual = dict(param)
            param_actual.pop("type", None)
            param_actual.pop("width", None)

            conv = lm.Convolution3dModule(
                **param_actual,
                activation=lbann.LeakyRelu,
                name=self.name+"_"+conv_name,
                bias=False,
                weights=conv_weights)
            setattr(self, conv_name, conv)

        # Create fully-connected layers
        fc_params = [
            {"size": 2048},
            {"size": 256},
            {"size": output_size},
        ]
        for i, param in enumerate(fc_params):
            fc_name ="fc"+str(i+1)
            fc = lm.FullyConnectedModule(
                **param,
                activation=lbann.LeakyRelu if i < len(fc_params)-1 else None,
                name=self.name+"_"+fc_name,
                weights=[Weights(initializer=lbann.GlorotUniformInitializer()),
                         Weights(initializer=lbann.ConstantInitializer(value=0.1))],
            )
            setattr(self, fc_name, fc)

    def forward(self, x):
        self.instance += 1

        def create_pooling(x, i, w):
            return lbann.Pooling(
                x, num_dims=3, has_vectors=False,
                pool_dims_i=3,
                pool_pads_i=1,
                pool_strides_i=2,
                pool_mode='average',
                name='{0}_pool{1}_instance{2}'.format(self.name,i,self.instance))

        def create_dropout(x, i):
            return lbann.Dropout(x, keep_prob=0.8,
                                 name='{0}_drop{1}_instance{2}'.format(self.name,i,self.instance))

        # Convolutional network
        i_conv = 1
        i_pool = 1
        for param in self.layer_params:
            if param["type"] == "conv":
                x = getattr(self, "conv{}".format(i_conv))(x)
                i_conv += 1

            else:
                x = create_pooling(x, i_pool, param["width"])
                i_pool += 1

        # Fully-connected layers
        x = create_dropout(x,1)
        x = self.fc1(x)
        x = create_dropout(x,2)
        x = self.fc2(x)
        x = create_dropout(x,3)
        x = self.fc3(x)

        return x

def create_data_reader(train_path, val_path, test_path):
    readerArgs = []
    for role, data_filename in [("train",    train_path),
                                ("validate", val_path),
                                ("test",     test_path)]:
        if not data_filename is None:
            readerArgs.append({"role": role, "data_filename": data_filename})

    readers = []
    for readerArg in readerArgs:
        reader = lp.lbann_pb2.Reader(
            name="numpy_npz_conduit_reader",
            shuffle=True,
            validation_percent=0,
            absolute_sample_count=0,
            percent_of_data_to_use=1.0,
            scaling_factor_int16=1.0,
            **readerArg)

        readers.append(reader)

    return lp.lbann_pb2.DataReader(reader=readers)

# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Construct and run the CosmoFlow network. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    '--partition', action='store', type=str,
    help='scheduler partition', metavar='NAME')
parser.add_argument(
    '--account', action='store', type=str,
    help='scheduler account', metavar='NAME')
parser.add_argument(
    '--experiment-dir', action='store', type=str,
    help='experiment directory', metavar='NAME')
parser.add_argument(
    "--learn-rate", action="store", default=0.0005, type=float,
    help="The initial learning-rate")
parser.add_argument(
        "--nodes", action="store", default=32, type=int,
        help="The number of nodes")
parser.add_argument(
        "--mini-batch-size", action="store", default=128, type=int,
        help="The mini-batch size")
parser.add_argument(
        "--epochs", action="store", default=130, type=int,
        help="The number of epochs")
parser.add_argument(
        "--output-size", action="store", default=4, type=int,
        help="Size of output tensor")
parser.add_argument(
        "--input-width", action="store", default=256, type=int,
        help="Width of input tensor")
for role, label, required in [("train", "training",   True),
                              ("val",   "validation", False),
                              ("test",  "test",       False)]:
    parser.add_argument(
            "--{}-path".format(role), type=str, required=required,
            help="Path to {} dataset".format(label), default=None)
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Input data
input = lbann.Input(io_buffer='partitioned',
                    target_mode='regression')
universes = lbann.Identity(input)
secrets = lbann.Identity(input)

# CosmoFlow
x = CosmoFlow(args.output_size,
              args.input_width).forward(universes)

# Loss function
loss = lbann.MeanSquaredError([x, secrets])

# Metrics
metrics = [lbann.Metric(loss, name="MSE", unit="")]

# Callbacks
callbacks = [
    lbann.CallbackPrint(),
    lbann.CallbackTimer(),
    lbann.CallbackPolyLearningRate(
        power=1.0,
        num_epochs=100, # TODO: Warn if args.epochs < 100
    ),
    lbann.CallbackGPUMemoryUsage(),
    lbann.CallbackDumpOutputs(
        directory="dump_acts/",
        layers="cosmoflow_module1_fc3_instance1 layer3",
        execution_modes="test"
    ),
    lbann.CallbackProfiler(skip_init=True)
]

# ----------------------------------
# Setup experiment
# ----------------------------------

# Setup model
model = lbann.Model(args.mini_batch_size,
                    args.epochs,
                    layers=lbann.traverse_layer_graph(input),
                    objective_function=loss,
                    metrics=metrics,
                    callbacks=callbacks)

# Setup optimizer
opt = lbann.Adam(learn_rate=args.learn_rate,
                 beta1=0.9,
                 beta2=0.99,
                 eps=1e-8)

# Setup data reader
data_reader_proto = create_data_reader(args.train_path,
                                       args.val_path,
                                       args.test_path)

# ----------------------------------
# Run experiment
# ----------------------------------
# Note: Use `lbann.run` instead for non-LC systems.

kwargs = {}
if args.partition: kwargs['partition'] = args.partition
if args.account: kwargs['account'] = args.account
if args.experiment_dir: kwargs['experiment_dir'] = args.experiment_dir

lbann.contrib.lc.launcher.run(model, data_reader_proto, opt,
                              lbann_args=" --use_data_store --preload_data_store",
                              job_name='lbann_cosmoflow',
                              nodes=args.nodes,
                              **kwargs)
