import math, os, sys
import lbann_pb2
import google.protobuf.text_format as txtf
pb = lbann_pb2.LbannPB()

# These two globals used to handle implicit activations in keras layers
prev_layer = ''
activations = {'relu' : 0, 'sigmoid' : 0,'softmax' : 0,  'tanh' : 0}
# This is a list of keras layers which do not exist as single layers in lbann, but can be constructed using multiple.
complex_layers = ['LSTM']

# This is the main driving function. Setups model parameters passed to it, and uses the keras model object to build the protobuf model
def keras_to_lbann(model, num_classes,
        name='directed_acyclic_graph_model', data_layout="data_parallel",
        block_size=256, epochs=20,
        batch_size=64, num_parallel_readers=0,
        procs_per_model=0, callbacks=['timer','print'], target='target'):
    # set user passed parameters (currently set once for entire model
    pb.model.name = name
    pb.model.data_layout = data_layout
    pb.model.mini_batch_size = batch_size
    pb.model.block_size = block_size
    pb.model.num_epochs = epochs
    pb.model.num_parallel_readers = num_parallel_readers
    pb.model.procs_per_model = procs_per_model

    if model.layers[0].name != 'input_1':
        l = pb.model.layer.add()
        l.name = model.input_names[0]
        exec('l.input.SetInParent()')
        l.input.io_buffer = "partitioned"
    setup_layers(model)
    # allow user to specify we need a reconstruciton target layer
    target_layer(model,target)
    setup_metrics(model)
    setup_obj_func(model)
    # Intention here is to eventually allow users to add lbann callbacks via this mechanism
    setup_callbacks(callbacks)
    out_model()
    # stop us from actually running in tensorflow.
    exit(0)
# Setup functions. Iterate through keras model components and add them to LBANN protobuf model
def setup_layers(model):
    for i in model.layers:
        layer_type = layer_type_to_str(i)
        nested_model = 'Model' in layer_type
        l = None
        global prev_layer
        explicit_activation = 'Activation' in layer_type #or 'activation' in prev_layer
        if (not nested_model and layer_type not in complex_layers) or explicit_activation:
            l = pb.model.layer.add()
            l.name = i.name
            l.parents = get_parents(i, model)
            prev_layer = l.name
        if 'Input' in layer_type:
            input(i,l)
        elif 'Conv' in layer_type and 'Transpose' not in layer_type:
            conv(i,l)
        elif 'Conv' in layer_type and 'Transpose' in layer_type:
            deconv(i,l)
        elif 'Dense' in layer_type:
            dense(i,l)
        elif 'Pool' in layer_type:
            pool(i,l)
        elif 'Dropout' in layer_type:
            dropout(i,l)
        elif 'Flatten' in layer_type or 'Reshape' in layer_type:
            reshape(i,l)
        elif 'BatchNormalization' in layer_type:
            batch_norm(i,l)
        elif 'Add' in layer_type:
            add(i,l)
        #elif 'LSTM' in layer_type:
        #    lstm(i,pb.model)
        elif nested_model:
            setup_layers(i)
        elif explicit_activation:
            activation(i,l)
        elif hasattr(i,'activation') and i.activation.__name__ and not explicit_activation:
            activation(i,l)
        else:
            print('NEED TO IMPLEMENT: ' + i.name)

def setup_metrics(model):
    for i in model.metrics:
        metric_type = standardize_input(i)
        if 'accuracy' in metric_type:
            metric = pb.model.metric.add()
            exec('metric.categorical_accuracy.SetInParent()')
        else:
            print("IMPLEMENT (metric): " + metric_type)


def setup_obj_func(model):

    obj_func = standardize_input(model.loss)
    if "crossentropy" in obj_func:
        pb_obj_func = pb.model.objective_function.cross_entropy.add()
    elif 'mse' in obj_func:
        pb_obj_func = pb.model.objective_function.mean_squared_error.add()
    else:
        print("IMPLEMENT (obj_func): " + obj_func)
    pb_obj_func = pb.model.objective_function.l2_weight_regularization.add()
    pb_obj_func.scale_factor = .0001


def setup_callbacks(callbacks):
    for i in callbacks:
        callback = pb.model.callback.add()
        exec('callback.' + i +'.SetInParent()')

# LAYERS
# IO layers
def input(keras_layer, pb_layer):
    exec('pb_layer.input.SetInParent()')
    pb_layer.input.io_buffer = "partitioned"

def target_layer(model, target):
    l = pb.model.layer.add()
    # This is making 2 big time assumptions -- 1) last layer in keras model feeds to target, 2) first input layer is what we want as parent.
    l.parents = prev_layer + ' ' + model.input_names[0]
    l.name = target
    exec('l.' + target + '.SetInParent()')

# Learning layers
def conv(keras_layer, pb_layer):
    exec('pb_layer.convolution.SetInParent()')
    pb_layer.convolution.has_vectors = True
    pb_layer.convolution.num_dims = keras_layer.rank
    pb_layer.convolution.conv_dims = keras_tuple_to_protobuf(keras_layer.kernel_size)
    pb_layer.convolution.num_output_channels = keras_layer.filters
    pb_layer.convolution.conv_strides = keras_tuple_to_protobuf(keras_layer.strides)
    pb_layer.convolution.conv_pads = convert_padding(keras_layer)
    if keras_layer.use_bias:
        pb_layer.convolution.has_bias = True

# deconv is so redundant. prolly a way to combine these
def deconv(keras_layer, pb_layer):
    exec('pb_layer.deconvolution.SetInParent()')
    pb_layer.deconvolution.has_vectors = True
    pb_layer.deconvolution.num_dims = keras_layer.rank
    pb_layer.deconvolution.conv_dims = keras_tuple_to_protobuf(keras_layer.kernel_size)
    pb_layer.deconvolution.num_output_channels = keras_layer.filters
    pb_layer.deconvolution.conv_strides = keras_tuple_to_protobuf(keras_layer.strides)
    pb_layer.deconvolution.conv_pads = convert_padding(keras_layer)
    if keras_layer.use_bias:
        pb_layer.deconvolution.has_bias = True

def dense(keras_layer, pb_layer):
    exec('pb_layer.fully_connected.SetInParent()')
    pb_layer.fully_connected.num_neurons = keras_layer.units
    if keras_layer.use_bias:
        pb_layer.fully_connected.has_bias = True

# Regularizer layers

def batch_norm(keras_layer, pb_layer):
    exec('pb_layer.batch_normalization.SetInParent()')
    pb_layer.batch_normalization.decay = keras_layer.momentum
    pb_layer.batch_normalization.epsilon = keras_layer.epsilon

def dropout(keras_layer, pb_layer):
    exec('pb_layer.dropout.SetInParent()')
    pb_layer.dropout.keep_prob = 1 - keras_layer.rate

# Transform Layers

def add(keras_layer, pb_layer):
     exec('pb_layer.sum.SetInParent()')

def reshape(keras_layer, pb_layer):
    exec('pb_layer.reshape.SetInParent()')
    if 'flatten' in keras_layer.name.lower():
        pb_layer.reshape.flatten = True
    else:
        pb_layer.reshape.num_dims = len(keras_layer.target_shape)
        pb_layer.reshape.dims = keras_tuple_to_protobuf(keras_layer.target_shape)

def pool(keras_layer, pb_layer):
    exec('pb_layer.pooling.SetInParent()')
    pb_layer.pooling.has_vectors = True
    pb_layer.pooling.num_dims = len(keras_layer.pool_size)
    pb_layer.pooling.pool_dims = keras_tuple_to_protobuf(keras_layer.pool_size)
    pb_layer.pooling.pool_strides = keras_tuple_to_protobuf(keras_layer.strides)
    pb_layer.pooling.pool_pads = convert_padding(keras_layer)
    if 'max' in keras_layer.name:
        pb_layer.pooling.pool_mode = 'max'
    elif 'average' in keras_layer.name:
        pb_layer.pooling.pool_mode = 'average'

# Activation layers (handles all)
def activation(keras_layer, pb_layer):
    # If a pb_layer is not passed that implies this is an implicit activation passed as parameter on a different layer
    if not pb_layer:
        name = keras_layer.activation.__name__
        global activations
        if name == 'linear':
            return
        if name in activations:
            activations[name] += 1
            l = pb.model.layer.add()
            l.name = name + '_' + str(activations[name])
            global prev_layer
            l.parents = prev_layer
            prev_layer = l.name
            exec('l.' + name + '.SetInParent()')
        else:
            print("IMPLEMENT ACTIVATION: ", name)
            return
    else:
        exec('pb_layer.' + keras_layer.activation.__name__ + '.SetInParent()')

# Complex layers. Currently not fleshed out. This section is for layers defined as single layers in keras that can be stiched together using multiple lbann layers.
def lstm(keras_layer, model):

    #l = pb.model.layer.add()
    #l.name = "forget_gate"
    #l.parents = get_parents(i._inbound_nodes[0])
    #global prev_layer
    #prev_layer = l.name
    #exec('l.fully_connected.SetInParent()')
    return 0

# Utils
def convert_padding(keras_layer):
    if 'pool' in keras_layer.name:
        temp_pad = int(keras_layer.pool_size[0])
        dims = len(keras_layer.pool_size)
    elif 'conv' in keras_layer.name:
        temp_pad = int(keras_layer.kernel_size[0])
        dims = keras_layer.rank
    if keras_layer.padding == 'same':
        pad = math.floor((temp_pad - 1)/2)
        return " ".join(str(i) for i in [pad]*dims)
    elif keras_layer.padding == 'valid':
         return " ".join(str(i) for i in [0]*dims)

def get_parents(layer, model):
    # This checks for implicit activation layer
    # Implicit activations do not get listed in keras as separate layers, but we need them in LBANN
    if prev_layer[:-2] in activations:
        return prev_layer
    # This is mostly for input layers, but if parents don't exist don't try to add them
    elif not layer._inbound_nodes:
        return ''
    # This is the standard case, adds parents according to list of parent layers held in keras layers
    else:
        parents = ''
        for node in layer._inbound_nodes:
            for parent in node.inbound_layers:
                parents += parent.name + ' '
        return parents

def layer_type_to_str(layer):
    return str(layer.__class__)

def standardize_input(func):
     if type(func) is str:
          return func
     else:
         return str(func.__name__)

def keras_tuple_to_protobuf(inp):
    return ' '.join(map(str,inp))

def out_model():
    out_file = sys.argv[0][:-2] + "prototext"
    with open(out_file, "w") as f:
        f.write(txtf.MessageToString(pb))
    print('Created LBANN prototext: ' + out_file + '. usage: "srun -n <mpi_rank> lbann --model=' + out_file +' --reader=<path to data reader --optimizer=<path to optimizer>"')
