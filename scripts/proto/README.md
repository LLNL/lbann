# LBANN Python Prototext Interface

This provides a convenient Python wrapper for writing and generating LBANN model
prototexts. The syntax is meant to be deliberately reminiscent of
[PyTorch](https://pytorch.org/). If you use that, it should be familiar.

This is still a work in progress, please open an issue if you find any problems
or have feature suggestions.

# Setup

* This requires Python 3. You may need to load the relevant module for your
environment to be set up right.
* You will need the Python protobuf module, which can be locally installed with
`pip3 install --user protobuf`.
* You will need a build of LBANN.

_Advanced users_: This requires the `lbann_pb2` Python module generated from
`lbann.proto` using the `protoc` compiler. The LBANN build process should do
this automatically, installing it to `LBANN_BUILD_DIR/install/share/python`.
This package uses some basic heuristics to attempt to locate this if
`lbann_pb2` is not in your default Python search path. If these fail, you
should manually set your Python path to include the directory with `lbann_pb2`
(e.g. by setting the `PYTHONPATH` environment variable).

# Use

This consists of two components, `lbann_proto` and `blocks`. `lbann_proto` is an
automatically generated interface to (most of) the components of the LBANN
prototext system, including all layers, callbacks, objective functions,
regularizers, and metrics. `blocks` consists of manually-curated higher-level
building blocks that are commonly used in neural networks.

## `lbann_proto`

The basic workflow is to declare your layers and other network components,
construct the network architecture (which can be an arbitrary DAG), and then
generate the model prototext. Once created, each layer instance is _callable_.
Calling it with another layer _l_ makes _l_ a parent of that layer. Writing the
final prototext is done with the `save_model` method.

### Example

This creates a very simple convolutional neural network. The inline comments
explain how things work. (Note: This is not a very good model, but you can run
it on MNIST data.)

```py
import lbann_proto as lp

dl = 'data_parallel'  # This is the data layout we will use.

# Create the layers.
# The first two arguments to every layer are always its name and data layout.
# The input layer has two outputs: images and labels. The first child will get
# the images output, the second child the labels. Multiple other layers will
# need the labels, so we use a split layer. We also create a split layer as the
# first child for convenience.
input_layer = lp.Input('data', dl, io_buffer='partitioned')
images = lp.Split('images', dl)
labels = lp.Split('labels', dl)
# Create a convolutional layer.
conv1 = lp.Convolution('conv1', dl,
                       num_dims=2,  # Operates on two-dimensional data.
                       num_output_channels=64,  # 64 output channels (filters).
                       conv_dims_i=5,  # Kernel dimensions, 5x5.
                       conv_pads_i=2,  # Padding size, 2 on every dimension.
                       conv_strides_i=2,  # Stride, 2 in every dimension.
                       has_bias=True)  # Layer has a bias.
# Create a batch normalization layer.
bn1 = lp.BatchNormalization('bn1', dl,
                            decay=0.9,  # Decay for the moving average.
                            epsilon=1e-5)  # Thing to avoid divide-by-0.
# Create a ReLU layer.
relu1 = lp.Relu('relu1', dl)  # No parameters.
# Create a pooling layer.
pool1 = lp.Pooling('pool1', dl,
                   num_dims=2,  # Operates on two-dimensional data.
                   pool_dims_i=3,  # Kernel dimensions, 3x3.
                   pool_pads_i=1,  # Padding size, 1 on every dimension.
                   pool_strides_i=2,  # Stride, 2 in every dimension.
                   pool_mode='max')  # Do max pooling.
# Create a fully-connected layer.
fc1 = lp.FullyConnected('fc1', dl,
                        num_neurons=10,  # Number of neurons in the layer.
                        has_bias=False)  # Layer does not have a bias.
# Create a softmax layer.
softmax = lp.Softmax('prob', dl)

# These layers are used for the objective function and metrics.
# Create a cross-entropy layer.
cross_entropy = lp.CrossEntropy('cross_entropy', dl)
# Create a top-1 categorical accuracy layer.
top1 = lp.CategoricalAccuracy('top1_accuracy', dl)
# Create a top-5 categorical accuracy layer.
top5 = lp.TopKCategoricalAccuracy('top5_accuracy', dl,
                                  k=5)  # Number of top places to consider.

# Construct the network architecture.
# Note the order matters for the children of the input layer!
images(input_layer)  # input_layer is now the parent of images.
labels(input_layer)  # input_layer is also the parent of labels.
relu1(bn1(conv1(images)))  # First apply conv1, then bn1, then relu1.
pool1(relu1)  # Pool after the ReLU.
fc1(pool1)  # Feed the output of the pooling layer to the FC layer.
softmax(fc1)  # Softmax takes the output of the FC layer.
# These layers have two inputs: the network predictions (softmax), and the
# ground-truth labels. This syntax is simply using one line to set both parents.
cross_entropy(softmax)(labels)
top1(softmax)(labels)
top5(softmax)(labels)

# Create the objective functions and metrics.
obj = lp.ObjectiveFunction(
    cross_entropy,  # The objective function uses the cross-entropy layer.
    # Also apply L2 weight regularization, with a scale factor of 1e-4.
    [lp.L2WeightRegularization(scale_factor=1e-4)])
# The metrics take the name to display, the associated layer, and a unit symbol.
top1_metric = lp.Metric('categorical accuracy', top1, '%')
top5_metric = lp.Metric('top-5 categorical accuracy', top5, '%')

# Create callbacks.
callbacks = [
    lp.CallbackPrint(),  # This prints basic information every epoch.
    lp.CallbackTimer()  # This prints timing information every epoch.
]

# Save the model to a prototext file.
lp.save_model(
    'test.prototext',  # Write to test.prototext.
    input_layer,  # Provide the first layer of the model (typically input).
    dl,  # The data layout metrics, etc. will use.
    256,  # The mini-batch size.
    10,  # The number of epochs to train for.
    obj,  # The objective function to train with.
    metrics=[top1_metric, top5_metric],  # Metrics associated with the model.
    callbacks=callbacks)  # Callbacks to add to the model.

```

### Layers, etc.

Right now, the best source for layers (or similar) and their associated
arguments is to check the `src/proto/lbann.proto` file. All fields present in
a message there are supported as keyword arguments in this API.

## Building blocks

This presently consists of a small number of blocks, which are essentially
compound layers consisting of multiple other layers. They have a similar
interface as above.

For example, instead of creating separate convolutional, batch normalization,
and ReLU layers as above, we could instead do:
```py
import blocks as b

conv1 = b.ConvBNRelu2d(
      'conv1',  # Name. The component layers will have names derived from this.
      dl,  # Data layout.
      64,  # Number of filters.
      5,  # Kernel size.
      stride=2,  # Stride.
      padding=2,  # Padding.
      bias=True)  # Has a bias.
```
