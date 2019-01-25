# LBANN Python Prototext Interface

This provides a convenient Python wrapper for writing and generating
LBANN model prototext files. The syntax is meant to be deliberately
reminiscent of [PyTorch](https://pytorch.org/). If you use that, it
should be familiar.

This is still a work in progress, so please open an issue if you find
any problems or have feature suggestions.

For more details about the LBANN/ONNX converter, see
[here](docs/onnx/README.md).

# Setup

Requirements:
* Python 3. You may need to load the relevant module for your
  environment to be set up right.
* The Python protobuf module, which can be locally installed with
  `pip3 install --user protobuf`.
* A build of LBANN.

_Advanced users_: This requires the `lbann_pb2` Python module
generated from `lbann.proto` using the `protoc` compiler. The LBANN
build process should do this automatically, installing it to
`LBANN_BUILD_DIR/install/share/python`.  This package uses some basic
heuristics to attempt to locate this if `lbann_pb2` is not in your
default Python search path. If these fail, you should manually set
your Python path to include the directory with `lbann_pb2` (e.g. by
setting the `PYTHONPATH` environment variable).

Run `pip3 install -e .` on this directory to install this package.

# Use

This consists of two components, `lbann_proto` and
`lbann_modules`. `lbann_proto` is an automatically generated interface
to (most of) the components of the LBANN prototext system, e.g. the
layers, weights, objective functions, metrics, and
callbacks. `lbann_modules` consists of manually-curated higher-level
building blocks that are commonly used in neural networks.

_Possible points of confusion_: LBANN constructs a static graph of
layers (specifically a DAG), as opposed to the dynamic execution
graphs that are supported in some other frameworks (e.g. PyTorch and
TensorFlow). This interface is just for building that graph, and does
not actually run the model. Also note that a module is different from
a layer: a layer is a single instance of an operator, whereas a module
creates multiple instances of a (set of) layers with the same
parameters.

## `lbann_proto`

Neural network model components:

* A `Layer` is a tensor operation, arranged within a directed acyclic
  graph. A layer will recieve input tensors from its parents and will
  send output tensor to its children. Once the layer graph has been
  constructed, it may be helpful to call `traverse_layer_graph`, which
  is a generator function that traverses the layer graph in a
  topological order.
* A `Weights` is a set of trainable parameters. They are typically
  associated with one or more layers. Their initial values are
  populated with an `Initializer`.
* The `ObjectiveFunction` is a mathematical expression that the
  optimization algorithm will attempt to minimize. It is made up of
  multiple `ObjectiveFunctionTerm`s, which are added up (possibly with
  scaling factors) to obtain the full objective function. There are
  currently two objective function terms:
    - `LayerTerm` gets its value from a `Layer`. The layer must output
      a scalar (tensor with one entry).
    - `L2WeightRegularization` gets its value by computing the L2 norm
      of the model weights.
* A `Metric` reports values to the user, which is helpful for
  evaluating the progress of training. They get the their values from
  layers, which must output scalars (tensors with one entry).
* A `Callback` performs some function at various points during
  training. They are helpful for performing advanced training
  techniques.

Once all model components have been constructed, they can be exported
to a prototext file with the `save_model` method.

### Example

A simple (and not very good) convolutional neural network for MNIST
data:

```py
import lbann_proto as lp

# ----------------------------------------------------------
# Construct layer graph.
# ----------------------------------------------------------
# Note: The first argument to every layer specifies its parents,
# i.e. the sources for its input tensors.

# Input data.
# Note: Order matters for the children of the input layer!
input = lp.Input(io_buffer='partitioned') # Interacts with data reader.
images = lp.Identity(input)     # NCHW image tensor.
labels = lp.Identity(input)     # One-hot vector.

# Simple convolutional network.
conv = lp.Convolution(
    images,
    num_dims=2,             # 2D convolution for NCHW tensors.
    num_output_channels=64, # I.e. number of filters.
    conv_dims_i=5,          # Convolution window size (64x3x5x5 kernel).
    conv_pads_i=2,          # Padding of 2 in every dimension.
    conv_strides_i=2,       # Stride of 2 in every dimension.
    has_bias=True)          # Channel-wise bias.
bn = lp.BatchNormalization(conv)
relu = lp.Relu(bn)
pool = lp.Pooling(
    relu,
    num_dims=2,         # 2D pooling (for NCHW tensors).
    pool_dims_i=3,      # 3x3 pooling window.
    pool_pads_i=1,      # Padding of 1 in every dimension.
    pool_strides_i=2,   # Stride of 2 in every dimension.
    pool_mode='max')    # Max pooling.
fc = lp.FullyConnected(pool,
                       num_neurons=10,  # Output size.
                       has_bias=False)  # Entry-wise bias.
softmax = lp.Softmax(fc)

# Compute values for objective function and metrics.
cross_entropy = lp.CrossEntropy([softmax, labels])
top1 = lp.CategoricalAccuracy([softmax, labels])
top5 = lp.TopKCategoricalAccuracy([softmax, labels], k=5)

# ----------------------------------------------------------
# Construct objective function, metrics, and callbacks.
# ----------------------------------------------------------

obj = lp.ObjectiveFunction([
    cross_entropy,
    lp.L2WeightRegularization(scale_factor=1e-4)])  # L2 weight regularization
])
metrics = [
    lp.Metric(top1, name='categorical accuracy', unit='%'),
    lp.Metric(top5, name='top-5 categorical accuracy', unit='%')
]
callbacks = [
    lp.CallbackPrint(), # Print basic information every epoch.
    lp.CallbackTimer()  # Print timing information every epoch.
]

# ----------------------------------------------------------
# Save the model to a prototext file.
# ----------------------------------------------------------

lp.save_model(
    'test.prototext',   # Write to test.prototext.
    256,                # Mini-batch size.
    10,                 # Number of epochs for training.
    layers=traverse_layer_graph(input), # Get all layers connected to input.
    objective_function=obj,
    metrics=metrics,
    callbacks=callbacks)

```

### Documentation

Right now, the best source for documentation on neural network
components (layers, weights, etc.) is in `src/proto/lbann.proto`. All
fields present in a message are supported as keyword arguments in this
API.

## `lbann_modules`

This presently consists of a small number of neural network modules,
which are patterns of layers that take an input layer to produce an
output layer. Once created, a `Module` is _callable_. Calling it with
an input layer will add the module's pattern to the layer graph and
will return the output layer. For example, instead of creating a
convolution layer as above, we could instead create a convolution
module:

```py
import lbann_modules as lm
conv_module = lm.Convolution2dModule(
    64,         # Number of output channels, i.e. number of filters.
    5,          # Convolution window size (64x3x5x5 kernel).
    stride=2,   # Padding of 2 in every dimension.
    padding=2,  # Stride of 2 in every dimension.
    bias=True)  # Channel-wise bias.
conv = conv_module(images)  # images is a Layer
```
