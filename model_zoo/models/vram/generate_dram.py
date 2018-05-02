#!/usr/bin/env python
import sys
import os
import subprocess
import functools

# Parameters
lbann_dir       = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip()
lbann_proto_dir = lbann_dir + '/src/proto/'
work_dir        = lbann_dir + '/model_zoo/models/vram'
template_proto  = lbann_dir + '/model_zoo/models/vram/dram_template.prototext'
output_proto    = lbann_dir + '/model_zoo/models/vram/dram.prototext'

# Convert a list into a space-separated string
def str_list(l):
    if isinstance(l, list):
        return ' '.join(str(i) for i in l)
    elif isinstance(l, str):
        return l
    else:
        raise TypeError('str_list expects a list or a string')

# Construct a new layer and add it to the model
def new_layer(model, name, parents, layer_type, device = ''):
    l = model.layer.add()
    l.name = name
    l.parents = str_list(parents)
    l.device_allocation = device
    exec('l.' + layer_type + '.SetInParent()')
    return l

# Construct a new set of weights and add it to the model
def new_weights(model, name, initializer = 'constant_initializer'):
    w = model.weights.add()
    w.name = name
    exec('w.' + initializer + '.SetInParent()')
    return w

def add_lstm(model, name, parent, size):

    # Forget gate
    forget_gate = name + '_forget_gate'
    fc1 = forget_gate + '_fc1'
    fc2 = forget_gate + '_fc2'
    linearity = forget_gate + '_linearity'
    l = new_layer(model, fc1, parent, 'fully_connected')
    l.fully_connected.num_neurons = size
    l.fully_connected.has_bias = True
    w = new_weights(model, fc1 + 'linearity', 'glorot_normal_initializer')
    l.weights = fc1 + 'linearity'
    l = new_layer(model, fc2, name, 'fully_connected')
    l.fully_connected.num_neurons = size
    l.fully_connected.has_bias = False
    w = new_weights(model, fc2 + 'linearity', 'glorot_normal_initializer')
    l.weights = fc2 + 'linearity'
    l = new_layer(model, linearity, [fc1, fc2], 'sum')
    l = new_layer(model, forget_gate, linearity, 'sigmoid')

    # Input gate
    input_gate = name + '_input_gate'
    fc1 = input_gate + '_fc1'
    fc2 = input_gate + '_fc2'
    linearity = input_gate + '_linearity'
    l = new_layer(model, fc1, parent, 'fully_connected')
    l.fully_connected.num_neurons = size
    l.fully_connected.has_bias = True
    w = new_weights(model, fc1 + 'linearity', 'glorot_normal_initializer')
    l.weights = fc1 + 'linearity'
    l = new_layer(model, fc2, name, 'fully_connected')
    l.fully_connected.num_neurons = size
    l.fully_connected.has_bias = False
    w = new_weights(model, fc2 + 'linearity', 'glorot_normal_initializer')
    l.weights = fc2 + 'linearity'
    l = new_layer(model, linearity, [fc1, fc2], 'sum')
    l = new_layer(model, input_gate, linearity, 'sigmoid')

    # Output gate
    output_gate = name + '_output_gate'
    fc1 = output_gate + '_fc1'
    fc2 = output_gate + '_fc2'
    linearity = output_gate + '_linearity'
    l = new_layer(model, fc1, parent, 'fully_connected')
    l.fully_connected.num_neurons = size
    l.fully_connected.has_bias = True
    w = new_weights(model, fc1 + 'linearity', 'glorot_normal_initializer')
    l.weights = fc1 + 'linearity'
    l = new_layer(model, fc2, name, 'fully_connected')
    l.fully_connected.num_neurons = size
    l.fully_connected.has_bias = False
    w = new_weights(model, fc2 + 'linearity', 'glorot_normal_initializer')
    l.weights = fc2 + 'linearity'
    l = new_layer(model, linearity, [fc1, fc2], 'sum')
    l = new_layer(model, output_gate, linearity, 'sigmoid')

    # Cell update
    cell_update = name + '_cell_update'
    fc1 = cell_update + '_fc1'
    fc2 = cell_update + '_fc2'
    linearity = cell_update + '_linearity'
    l = new_layer(model, fc1, parent, 'fully_connected')
    l.fully_connected.num_neurons = size
    l.fully_connected.has_bias = True
    w = new_weights(model, fc1 + 'linearity', 'glorot_normal_initializer')
    l.weights = fc1 + 'linearity'
    l = new_layer(model, fc2, name, 'fully_connected')
    l.fully_connected.num_neurons = size
    l.fully_connected.has_bias = False
    w = new_weights(model, fc2 + 'linearity', 'glorot_normal_initializer')
    l.weights = fc2 + 'linearity'
    l = new_layer(model, linearity, [fc1, fc2], 'sum')
    l = new_layer(model, cell_update, linearity, 'tanh')

    # Cell state
    cell_state = name + '_cell_state'
    history = cell_state + '_history'
    update = cell_state + '_update'
    raw = cell_state + '_raw'
    nonlinearity = cell_state + '_nonlinearity'
    l = new_layer(model, history, [forget_gate, cell_state], 'hadamard')
    l = new_layer(model, update, [input_gate, cell_update], 'hadamard')
    l = new_layer(model, raw, [history, update], 'sum')
    l = new_layer(model, cell_state, raw, 'reshape')
    l.reshape.num_dims = 1
    l.reshape.dims = str(size)
    l = new_layer(model, nonlinearity, cell_state, 'tanh')

    # Output
    raw = name + '_raw'
    l = new_layer(model, raw, [output_gate, nonlinearity], 'hadamard')
    l = new_layer(model, name, raw, 'reshape')
    l.reshape.num_dims = 1
    l.reshape.dims = str(size)

# Configure a prototext model (e.g. add layers)
def configure_model(model):

    # Data and labels
    data_dims = [1, 227, 227]
    label_dims = [1000]
    data_size = functools.reduce(lambda x, y: x*y, data_dims)
    label_size = functools.reduce(lambda x, y: x*y, label_dims)
    slice_points = [0, data_size, data_size + label_size]
    l = new_layer(model, 'input_concat', '', 'repeated_input')
    l.repeated_input.num_steps = model.recurrent.unroll_depth
    l = new_layer(model, 'input', 'input_concat', 'slice')
    l.children = 'data label'
    l.slice.slice_points = str_list(slice_points)
    l = new_layer(model, 'data', 'input', 'reshape')
    l.reshape.num_dims = len(data_dims)
    l.reshape.dims = str_list(data_dims)
    l = new_layer(model, 'label', 'input', 'reshape')
    l.reshape.num_dims = len(label_dims)
    l.reshape.dims = str_list(label_dims)

    # Useful constants
    l = new_layer(model, 'zero1', '', 'constant', 'cpu')
    l.constant.value = 0.0
    l.constant.num_neurons = '1'
    l = new_layer(model, 'zero3', '', 'constant', 'cpu')
    l.constant.value = 0.0
    l.constant.num_neurons = '3'
    l = new_layer(model, 'one1', '', 'constant', 'cpu')
    l.constant.value = 1.0
    l.constant.num_neurons = '1'

    # Glimpse position
    num_locs = 32
    locs = map(lambda i: 2.0 * i / num_locs - 1.0, range(num_locs))
    l = new_layer(model, 'scaled_locy', 'locy one1', 'sum', 'cpu')
    l.sum.scaling_factors = '0.5 0.5'
    l = new_layer(model, 'scaled_locx', 'locx one1', 'sum', 'cpu')
    l.sum.scaling_factors = '0.5 0.5'
    l = new_layer(model,
                  'scaled_loc',
                  'zero1 scaled_locy scaled_locx',
                  'concatenation',
                  'cpu')

    # Extract glimpse data
    crop1_dims = [3, 32, 32]
    crop2_dims = [3, 64, 64]
    crop3_dims = [3, 128, 128]
    l = new_layer(model, 'glimpse1', 'data scaled_loc', 'crop')
    l.crop.dims = str_list(crop1_dims)
    l = new_layer(model, 'crop2', 'data scaled_loc', 'crop')
    l.crop.dims = str_list(crop2_dims)
    l = new_layer(model, 'glimpse2', 'crop2', 'pooling')
    l.pooling.num_dims = 2
    l.pooling.pool_dims_i = 2
    l.pooling.pool_strides_i = l.pooling.pool_dims_i
    l.pooling.pool_mode = 'average'
    l = new_layer(model, 'crop3', 'data scaled_loc', 'crop')
    l.crop.dims = str_list(crop3_dims)
    l = new_layer(model, 'glimpse3', 'crop3', 'pooling')
    l.pooling.num_dims = 2
    l.pooling.pool_dims_i = 4
    l.pooling.pool_strides_i = l.pooling.pool_dims_i
    l.pooling.pool_mode = 'average'

    # Glimpse network
    l = new_layer(model, 'glimpse1_conv1', 'glimpse1', 'convolution')
    l.convolution.num_dims = 2
    l.convolution.num_output_channels = 32
    l.convolution.conv_dims_i = 3
    l.convolution.conv_strides_i = 1
    l.convolution.has_bias = True
    l = new_layer(model, 'glimpse1_relu1', 'glimpse1_conv1', 'relu')
    l = new_layer(model, 'glimpse1_conv2', 'glimpse1_relu1', 'convolution')
    l.convolution.num_dims = 2
    l.convolution.num_output_channels = 64
    l.convolution.conv_dims_i = 3
    l.convolution.conv_strides_i = 1
    l.convolution.has_bias = True
    l = new_layer(model, 'glimpse1_relu2', 'glimpse1_conv2', 'relu')
    l = new_layer(model, 'glimpse1_conv3', 'glimpse1_relu2', 'convolution')
    l.convolution.num_dims = 2
    l.convolution.num_output_channels = 128
    l.convolution.conv_dims_i = 3
    l.convolution.conv_strides_i = 1
    l.convolution.has_bias = True
    l = new_layer(model, 'glimpse1_relu3', 'glimpse1_conv3', 'relu')
    l = new_layer(model, 'glimpse1_pool', 'glimpse1_relu3', 'pooling')
    l.pooling.num_dims = 2
    l.pooling.pool_dims_i = 26
    l.pooling.pool_strides_i = 1
    l.pooling.pool_mode = 'average'
    l = new_layer(model, 'glimpse2_conv1', 'glimpse2', 'convolution')
    l.convolution.num_dims = 2
    l.convolution.num_output_channels = 32
    l.convolution.conv_dims_i = 3
    l.convolution.conv_strides_i = 1
    l.convolution.has_bias = True
    l = new_layer(model, 'glimpse2_relu1', 'glimpse2_conv1', 'relu')
    l = new_layer(model, 'glimpse2_conv2', 'glimpse2_relu1', 'convolution')
    l.convolution.num_dims = 2
    l.convolution.num_output_channels = 64
    l.convolution.conv_dims_i = 3
    l.convolution.conv_strides_i = 1
    l.convolution.has_bias = True
    l = new_layer(model, 'glimpse2_relu2', 'glimpse2_conv2', 'relu')
    l = new_layer(model, 'glimpse2_conv3', 'glimpse2_relu2', 'convolution')
    l.convolution.num_dims = 2
    l.convolution.num_output_channels = 128
    l.convolution.conv_dims_i = 3
    l.convolution.conv_strides_i = 1
    l.convolution.has_bias = True
    l = new_layer(model, 'glimpse2_relu3', 'glimpse2_conv3', 'relu')
    l = new_layer(model, 'glimpse2_pool', 'glimpse2_relu3', 'pooling')
    l.pooling.num_dims = 2
    l.pooling.pool_dims_i = 26
    l.pooling.pool_strides_i = 1
    l.pooling.pool_mode = 'average'
    l = new_layer(model, 'glimpse3_conv1', 'glimpse3', 'convolution')
    l.convolution.num_dims = 2
    l.convolution.num_output_channels = 32
    l.convolution.conv_dims_i = 3
    l.convolution.conv_strides_i = 1
    l.convolution.has_bias = True
    l = new_layer(model, 'glimpse3_relu1', 'glimpse3_conv1', 'relu')
    l = new_layer(model, 'glimpse3_conv2', 'glimpse3_relu1', 'convolution')
    l.convolution.num_dims = 2
    l.convolution.num_output_channels = 64
    l.convolution.conv_dims_i = 3
    l.convolution.conv_strides_i = 1
    l.convolution.has_bias = True
    l = new_layer(model, 'glimpse3_relu2', 'glimpse3_conv2', 'relu')
    l = new_layer(model, 'glimpse3_conv3', 'glimpse3_relu2', 'convolution')
    l.convolution.num_dims = 2
    l.convolution.num_output_channels = 128
    l.convolution.conv_dims_i = 3
    l.convolution.conv_strides_i = 1
    l.convolution.has_bias = True
    l = new_layer(model, 'glimpse3_relu3', 'glimpse3_conv3', 'relu')
    l = new_layer(model, 'glimpse3_pool', 'glimpse3_relu3', 'pooling')
    l.pooling.num_dims = 2
    l.pooling.pool_dims_i = 26
    l.pooling.pool_strides_i = 1
    l.pooling.pool_mode = 'average'
    l = new_layer(model,
                  'glimpse',
                  'glimpse1_pool glimpse2_pool glimpse3_pool',
                  'concatenation')
    
    # Recurrent network
    add_lstm(model, 'lstm1', 'glimpse', 128)
    add_lstm(model, 'lstm2', 'lstm1', 128)
    
    # Emission network
    l = new_layer(model, 'emission_y_fc', 'lstm2', 'fully_connected')
    l.fully_connected.num_neurons = num_locs
    l.fully_connected.has_bias = False
    l = new_layer(model, 'emission_x_fc', 'lstm2', 'fully_connected')
    l.fully_connected.num_neurons = num_locs
    l.fully_connected.has_bias = False
    l = new_layer(model, 'emission_y_probs', 'emission_y_fc', 'softmax')
    l = new_layer(model, 'emission_x_probs', 'emission_x_fc', 'softmax')
    l = new_layer(model, 'emission_y', 'emission_y_probs', 'categorical_random')
    l = new_layer(model, 'emission_x', 'emission_x_probs', 'categorical_random')
    l = new_layer(model, 'locy', 'emission_y', 'discrete_random')
    l.discrete_random.values = str_list(locs)
    l.discrete_random.dims = '1'
    l = new_layer(model, 'locx', 'emission_x', 'discrete_random')
    l.discrete_random.values = str_list(locs)
    l.discrete_random.dims = '1'

    # Classification network
    l = new_layer(model, 'classification_fc', 'lstm1', 'fully_connected')
    l.fully_connected.num_neurons = label_size
    l.fully_connected.has_bias = False
    l = new_layer(model, 'classification_probs', 'classification_fc', 'softmax')
    
    # Objective function
    l = new_layer(model,
                  'classification_prob_full',
                  'classification_probs label',
                  'hadamard')
    l = new_layer(model,
                  'classification_prob',
                  'classification_prob_full',
                  'reduction')
    l.reduction.mode = 'sum'
    l = new_layer(model,
                  'emission_y_prob_full',
                  'emission_y_probs emission_y',
                  'hadamard')
    l = new_layer(model,
                  'emission_y_prob',
                  'emission_y_prob_full',
                  'reduction')
    l.reduction.mode = 'sum'
    l = new_layer(model,
                  'emission_x_prob_full',
                  'emission_x_probs emission_x',
                  'hadamard')
    l = new_layer(model,
                  'emission_x_prob',
                  'emission_x_prob_full',
                  'reduction')
    l.reduction.mode = 'sum'
    l = new_layer(model,
                  'prob',
                  'classification_prob emission_y_prob emission_x_prob',
                  'hadamard')
    l = new_layer(model, 'obj', 'prob', 'log')
    l = new_layer(model, 'obj_eval', 'obj', 'evaluation')

    # Accuracy
    l = new_layer(model, 'prediction', 'classification_probs', 'categorical_random', 'cpu')
    l = new_layer(model, 'accuracy_full', 'label prediction', 'hadamard', 'cpu')
    l = new_layer(model, 'accuracy', 'accuracy_full', 'reduction')
    l.reduction.mode = 'sum'
    l = new_layer(model, 'accuracy_eval', 'accuracy', 'evaluation')
    
if __name__ == "__main__":

    # Make sure protobuf Python implementation is built
    host = subprocess.check_output('hostname').strip('\n1234567890')
    protoc = lbann_dir + '/build/gnu.' + host + '.llnl.gov/install/bin/protoc'
    proto_python_dir = lbann_dir + '/build/gnu.' + host + '.llnl.gov/protobuf/src/python'
    os.putenv('PROTOC', protoc)
    subprocess.call('cd ' + proto_python_dir + '; '
                    + sys.executable + ' '
                    + proto_python_dir + '/setup.py build',
                    shell=True)
    sys.path.append(proto_python_dir)
    import google.protobuf.text_format as txtf

    # Compile LBANN protobuf
    subprocess.call([protoc,
                     '-I=' + lbann_proto_dir,
                     '--python_out=' + work_dir,
                     lbann_proto_dir + '/lbann.proto'])
    sys.path.append(work_dir)
    global lbann_pb2
    import lbann_pb2

    # Load template prototext
    with open(template_proto, 'r') as f:
        pb = txtf.Merge(f.read(), lbann_pb2.LbannPB())

    # Configure prototext model
    configure_model(pb.model)

    # Export prototext
    with open(output_proto, 'w') as f:
        f.write(txtf.MessageToString(pb))
    
