#!/usr/bin/env python
import sys
import os
import subprocess
import functools
import collections

# Parameters
lbann_dir       = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip()
lbann_proto_dir = lbann_dir + "/src/proto/"
work_dir        = lbann_dir + "/model_zoo/models/vram"
template_proto  = lbann_dir + "/model_zoo/models/vram/dram_template.prototext"
output_proto    = lbann_dir + "/model_zoo/models/vram/dram.prototext"

# Convert a list into a space-separated string
def str_list(l):
    if isinstance(l, str):
        return l
    else:
        return " ".join(str(i) for i in l)

# Construct a new layer and add it to the model
def new_layer(model, name, parents, layer_type, device = "", weights = []):
    if not isinstance(parents, collections.Iterable):
        return new_layer(model, name, [parents], layer_type, device, weights)
    if not isinstance(weights, collections.Iterable):
        return new_layer(model, name, parents, layer_type, device, [weights])
    l = model.layer.add()
    l.name = name
    l.parents = str_list(map(lambda l : l.name, parents))
    exec("l." + layer_type + ".SetInParent()")
    l.weights = str_list(map(lambda w : w.name, weights))
    l.device_allocation = device
    return l

# Construct a new set of weights and add it to the model
def new_weights(model, name, initializer = ""):
    w = model.weights.add()
    w.name = name
    if initializer:
        exec("w." + initializer + ".SetInParent()")
    return w

class FullyConnectedCell:

    name = ""
    size = 0
    model = None
    has_bias = False
    activation = None
    weights = []
    step = -1

    def __init__(self, name, size, model,
                 activation = None, initializer = "constant_initializer", has_bias = True):
        self.name = name
        self.size = size
        self.model = model
        self.has_bias = has_bias
        self.activation = activation

        # Initialize weights
        self.weights = [new_weights(model, name + "_linearity", initializer),
                        new_weights(model, name + "_bias", "constant_initializer")]

    def __call__(self, parent):
        self.step += 1
        fc = new_layer(self.model, "%s_fc_step%d" % (self.name, self.step),
                       parent, "fully_connected", "" ,self.weights)
        fc.fully_connected.num_neurons = self.size
        fc.fully_connected.has_bias = self.has_bias
        if self.activation:
           act = new_layer(self.model,
                           "%s_step%d" % (self.name, self.step),
                           fc, self.activation)
           return act
        else:
            fc.name = "%s_step%d" % (self.name, self.step)
            return fc

class ConvolutionCell:

    name = ""
    num_output_channels = 0
    num_dims = 0
    conv_dim = 0
    conv_stride = 0
    conv_pad = 0
    model = None
    has_bias = False
    activation = None
    weights = []
    step = -1

    def __init__(self, name, num_output_channels,
                 num_dims, conv_dim, conv_stride, conv_pad,
                 model,
                 activation = None,
                 initializer = "constant_initializer",
                 has_bias = True):
        self.name = name
        self.num_output_channels = num_output_channels
        self.num_dims = num_dims
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad
        self.model = model
        self.has_bias = has_bias
        self.activation = activation

        # Initialize weights
        self.weights = [new_weights(model, name + "_kernel", initializer),
                        new_weights(model, name + "_bias", "constant_initializer")]

    def __call__(self, parent):
        self.step += 1
        conv = new_layer(self.model, "%s_conv_step%d" % (self.name, self.step),
                         parent, "convolution", "", self.weights)
        conv.convolution.num_output_channels = self.num_output_channels
        conv.convolution.num_dims = self.num_dims
        conv.convolution.conv_dims_i = self.conv_dim
        conv.convolution.conv_strides_i = self.conv_stride
        conv.convolution.conv_pads_i = self.conv_pad
        conv.convolution.has_bias = self.has_bias
        if self.activation:
           act = new_layer(self.model,
                           "%s_step%d" % (self.name, self.step),
                           conv, self.activation)
           return act
        else:
            conv.name = "%s_step%d" % (self.name, self.step)
            return conv

# Uses reLU activations
class LstmCell:

    name = ""
    size = 0
    model  = None
    step = -1
    outputs = []
    cells = []

    # Fully-connected layers
    forget_fc = None
    input_fc = None
    output_fc = None
    cell_fc = None
    
    def __init__(self, name, size, model):
        self.name = name
        self.size = size
        self.model = model

        # Fully-connected layers
        self.forget_gate = FullyConnectedCell(name + "_forget_gate_fc", size, model,
                                              "sigmoid", "glorot_normal_initializer", True)
        self.input_gate = FullyConnectedCell(name + "_input_gate_fc", size, model,
                                             "sigmoid", "glorot_normal_initializer", True)
        self.output_gate = FullyConnectedCell(name + "_output_gate_fc", size, model,
                                              "sigmoid", "glorot_normal_initializer", True)
        self.cell_update = FullyConnectedCell(name + "_cell_update_fc", size, model,
                                              "relu", "he_normal_initializer", True)

        # Initial state
        self.outputs = [new_layer(model, name + "_output_init", [], "constant")]
        self.outputs[0].constant.num_neurons = str(size)
        self.cells = [new_layer(model, name + "_cell_init", [], "constant")]
        self.cells[0].constant.num_neurons = str(size)

    def __call__(self, parent):
        self.step += 1

        # LSTM input state is from parent layer and previous output
        input_state = new_layer(self.model,
                                "%s_input_state_step%d" % (self.name, self.step),
                                [parent, self.outputs[-1]],
                                "concatenation")

        # Gating units
        f = self.forget_gate(input_state)
        i = self.input_gate(input_state)
        o = self.output_gate(input_state)

        # Cell state
        c = self.cell_update(input_state)
        cell_forget = new_layer(self.model,
                                "%s_cell_forget_step%d" % (self.name, self.step),
                                [f, self.cells[-1]], "hadamard")
        cell_input = new_layer(self.model,
                               "%s_cell_input_step%d" % (self.name, self.step),
                               [i, c], "hadamard")
        self.cells.append(new_layer(self.model,
                                    "%s_cell_step%d" % (self.name, self.step),
                                    [cell_forget, cell_input],
                                    "sum"))

        # Output
        act = new_layer(self.model,
                        "%s_cell_activation_step%d" % (self.name, self.step),
                        self.cells[-1], "relu")
        self.outputs.append(new_layer(self.model,
                                      "%s_step%d" % (self.name, self.step),
                                      [o, act], "hadamard"))
        return self.outputs[-1]

# Configure a prototext model (e.g. add layers)
def configure_model(model):

    # Model parameters
    unroll_depth = 4
    image_dims = [3, 227, 227]
    label_dims = [1000]
    hidden_size = 128   # RNN state size
    num_locs = 32

    # Initialize input
    data = new_layer(model, "data", [], "input", "cpu")
    data.input.io_buffer = "partitioned"
    image = new_layer(model, "image", data, "split")
    label = new_layer(model, "label", data, "split")
    data.children = str_list([image.name, label.name])

    # Initialize useful constants
    zero1 = new_layer(model, "zero1", [], "constant", "cpu")
    zero1.constant.value = 0.0
    zero1.constant.num_neurons = str_list([1])
    zero3 = new_layer(model, "zero3", [], "constant", "cpu")
    zero3.constant.value = 0.0
    zero3.constant.num_neurons = str_list([3])
    one3 = new_layer(model, "one3", [], "constant", "cpu")
    one3.constant.value = 1.0
    one3.constant.num_neurons = str_list([3])

    # Glimpse network components
    glimpse1_conv1 = ConvolutionCell("glimpse1_conv1", 32, 2, 3, 1, 1,
                                     model, "relu", "he_normal_initializer")
    glimpse1_conv2 = ConvolutionCell("glimpse1_conv2", 64, 2, 3, 1, 1,
                                     model, "relu", "he_normal_initializer")
    glimpse1_conv3 = ConvolutionCell("glimpse1_conv3", 128, 2, 3, 1, 1,
                                     model, "relu", "he_normal_initializer")
    glimpse2_conv1 = ConvolutionCell("glimpse2_conv1", 32, 2, 3, 1, 1,
                                     model, "relu", "he_normal_initializer")
    glimpse2_conv2 = ConvolutionCell("glimpse2_conv2", 64, 2, 3, 1, 1,
                                     model, "relu", "he_normal_initializer")
    glimpse2_conv3 = ConvolutionCell("glimpse2_conv3", 128, 2, 3, 1, 1,
                                     model, "relu", "he_normal_initializer")
    glimpse3_conv1 = ConvolutionCell("glimpse3_conv1", 32, 2, 3, 1, 1,
                                     model, "relu", "he_normal_initializer")
    glimpse3_conv2 = ConvolutionCell("glimpse3_conv2", 64, 2, 3, 1, 1,
                                     model, "relu", "he_normal_initializer")
    glimpse3_conv3 = ConvolutionCell("glimpse3_conv3", 128, 2, 3, 1, 1,
                                     model, "relu", "he_normal_initializer")

    # Recurrent network components
    lstm1 = LstmCell("lstm1", hidden_size, model)
    lstm2 = LstmCell("lstm2", hidden_size, model)

    # Location network components
    loc_list = map(lambda i: 2.0 * i / num_locs - 1.0, range(num_locs))
    loc = zero3
    locx_network = FullyConnectedCell("locx_prob", num_locs, model,
                                   "softmax", "glorot_normal_initializer", False)
    locy_network = FullyConnectedCell("locy_prob", num_locs, model,
                                   "softmax", "glorot_normal_initializer", False)

    # Classification network components
    class_network = FullyConnectedCell("class_prob", label_dims[0], model,
                                    "softmax", "glorot_normal_initializer", False)
    
    # Construct unrolled model
    for step in range(unroll_depth):

        # Extract crops and resize
        scaled_loc = new_layer(model, "loc_scaled_step%d" % step,
                               [loc, one3], "weighted_sum", "cpu")
        scaled_loc.weighted_sum.scaling_factors = str_list([0.5, 0.5])
        crop1 = new_layer(model, "crop1_step%d" % step,
                          [image, scaled_loc], "crop", "cpu")
        crop1.crop.dims = str_list([3, 32, 32])
        crop2 = new_layer(model, "crop2_step%d" % step,
                          [image, scaled_loc], "crop", "cpu")
        crop2.crop.dims = str_list([3, 64, 64])
        crop2 = new_layer(model, "crop2_resized_step%d" % step, crop2, "pooling")
        crop2.pooling.num_dims = 2
        crop2.pooling.pool_dims_i = 2
        crop2.pooling.pool_strides_i = crop2.pooling.pool_dims_i
        crop2.pooling.pool_mode = "average"
        crop3 = new_layer(model, "crop3_step%d" % step,
                          [image, scaled_loc], "crop", "cpu")
        crop3.crop.dims = str_list([3, 128, 128])
        crop3 = new_layer(model, "crop3_resized_step%d" % step, crop3, "pooling")
        crop3.pooling.num_dims = 2
        crop3.pooling.pool_dims_i = 4
        crop3.pooling.pool_strides_i = crop3.pooling.pool_dims_i
        crop3.pooling.pool_mode = "average"

        # Glimpse networks
        glimpse1 = glimpse1_conv1(crop1)
        glimpse1 = glimpse1_conv2(glimpse1)
        glimpse1 = glimpse1_conv3(glimpse1)
        glimpse1 = new_layer(model, "glimpse1_step%d" % step, glimpse1, "pooling")
        glimpse1.pooling.num_dims = 2
        glimpse1.pooling.pool_dims_i = 32
        glimpse1.pooling.pool_strides_i = glimpse1.pooling.pool_dims_i
        glimpse1.pooling.pool_mode = "average"
        glimpse2 = glimpse2_conv1(crop2)
        glimpse2 = glimpse2_conv2(glimpse2)
        glimpse2 = glimpse2_conv3(glimpse2)
        glimpse2 = new_layer(model, "glimpse2_step%d" % step, glimpse2, "pooling")
        glimpse2.pooling.num_dims = 2
        glimpse2.pooling.pool_dims_i = 32
        glimpse2.pooling.pool_strides_i = glimpse2.pooling.pool_dims_i
        glimpse2.pooling.pool_mode = "average"
        glimpse3 = glimpse3_conv1(crop3)
        glimpse3 = glimpse3_conv2(glimpse3)
        glimpse3 = glimpse3_conv3(glimpse3)
        glimpse3 = new_layer(model, "glimpse3_step%d" % step, glimpse3, "pooling")
        glimpse3.pooling.num_dims = 2
        glimpse3.pooling.pool_dims_i = 32
        glimpse3.pooling.pool_strides_i = glimpse3.pooling.pool_dims_i
        glimpse3.pooling.pool_mode = "average"
        glimpse = new_layer(model, "glimpse_step%d" % step, 
                            [glimpse1, glimpse2, glimpse3], "concatenation")
        glimpse = new_layer(model, "glimpse_flat_step%d" % step, 
                            glimpse, "reshape")
        glimpse.reshape.num_dims = 1
        glimpse.reshape.dims = str_list([128 * 3])
        
        # Recurrent network
        h1 = lstm1(glimpse)
        h2 = lstm2(h1)

        # Location network
        locx_prob = locx_network(h2)
        locx_onehot = new_layer(model, "locx_onehot_step%d" % step,
                                locx_prob, "categorical_random", "cpu")
        locx = new_layer(model, "locx_step%d" % step,
                         locx_onehot, "discrete_random", "cpu")
        locx.discrete_random.values = str_list(loc_list)
        locx.discrete_random.dims = str_list([1])
        locy_prob = locy_network(h2)
        locy_onehot = new_layer(model, "locy_onehot_step%d" % step,
                                locy_prob, "categorical_random", "cpu")
        locy = new_layer(model, "locy_step%d" % step,
                         locy_onehot, "discrete_random", "cpu")
        locy.discrete_random.values = str_list(loc_list)
        locy.discrete_random.dims = str_list([1])
        loc = new_layer(model, "loc_step%d" % (step+1),
                        [zero1, locy, locx], "concatenation", "cpu")

        # Classification network
        class_prob = class_network(h1)

        # Categorical accuracy
        acc1 = new_layer(model, "top1_accuracy_step%d" % step,
                         [class_prob, label], "top_k_categorical_accuracy")
        acc1.top_k_categorical_accuracy.k = 1
        acc5 = new_layer(model, "top5_accuracy_step%d" % step,
                         [class_prob, label], "top_k_categorical_accuracy")
        acc5.top_k_categorical_accuracy.k = 5
        met = model.metric.add()
        met.layer_metric.name = "categorical accuracy (step %d)" % step
        met.layer_metric.layer = acc1.name
        met.layer_metric.unit = "%"
        met = model.metric.add()
        met.layer_metric.name = "top-5 categorical accuracy (step %d)" % step
        met.layer_metric.layer = acc5.name
        met.layer_metric.unit = "%"
        
        # Objective function
        class_obj = new_layer(model, "classification_cross_entropy_step%d" % step,
                              [class_prob, label], "cross_entropy")
        locx_obj = new_layer(model, "locx_cross_entropy_step%d" % step,
                             [locx_prob, locx_onehot], "cross_entropy")
        locy_obj = new_layer(model, "locy_cross_entropy_step%d" % step,
                             [locy_prob, locy_onehot], "cross_entropy")
        obj = model.objective_function.layer_term.add()
        obj.scale_factor = 1.0
        obj.layer = class_obj.name
        obj = model.objective_function.layer_term.add()
        obj.scale_factor = 1.0
        obj.layer = locx_obj.name
        obj = model.objective_function.layer_term.add()
        obj.scale_factor = 1.0
        obj.layer = locy_obj.name
        
    
if __name__ == "__main__":

    # Make sure protobuf Python implementation is built
    host = subprocess.check_output("hostname").strip("\n1234567890")
    protoc = lbann_dir + "/build/gnu.Release." + host + ".llnl.gov/install/bin/protoc"
    proto_python_dir = lbann_dir + "/build/gnu.Release." + host + ".llnl.gov/protobuf/src/python"
    os.putenv("PROTOC", protoc)
    subprocess.call("cd " + proto_python_dir + "; "
                    + sys.executable + " "
                    + proto_python_dir + "/setup.py build",
                    shell=True)
    sys.path.append(proto_python_dir)
    import google.protobuf.text_format as txtf

    # Compile LBANN protobuf
    subprocess.call([protoc,
                     "-I=" + lbann_proto_dir,
                     "--python_out=" + work_dir,
                     lbann_proto_dir + "/lbann.proto"])
    sys.path.append(work_dir)
    global lbann_pb2
    import lbann_pb2

    # Load template prototext
    with open(template_proto, "r") as f:
        pb = txtf.Merge(f.read(), lbann_pb2.LbannPB())

    # Configure prototext model
    configure_model(pb.model)

    # Export prototext
    with open(output_proto, "w") as f:
        f.write(txtf.MessageToString(pb))
