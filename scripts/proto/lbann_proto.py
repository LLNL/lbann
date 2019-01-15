"""Generate prototexts for LBANN models."""

import google.protobuf.text_format
import lbann_pb2
#from . import lbann_pb2
import collections

def _add_to_module_namespace(stuff):
    """Add stuff to the module namespace.

    stuff is a dict, keys will be the name.

    """
    g = globals()
    for k, v in stuff.items():
        g[k] = v

class Layer:
    """Base class for layers."""

    num_layers = 0  # Static counter, used for default layer names

    def __init__(self, name='', inputs=[], outputs=[],
                 weights=[], data_layout=''):
        self.name = name if name else 'layer{0}'.format(Layer.num_layers)
        self.inputs = (inputs
                       if isinstance(inputs, (list,))
                       else [inputs])
        self.outputs = (outputs
                        if isinstance(outputs, (list,))
                        else [outputs])
        self.weights = (weights
                        if isinstance(weights, (list,))
                        else [weights])
        self.data_layout = data_layout
        Layer.num_layers += 1

    def export_proto(self):
        """Construct protobuf message for the layer."""
        proto = lbann_pb2.Layer()
        proto.parents = ' '.join([l.name for l in self.inputs])
        proto.children = ' '.join([l.name for l in self.outputs])
        proto.weights = ' '.join([w.name for w in self.weights])
        proto.name = self.name
        proto.data_layout = self.data_layout
        return proto

    def add_input(self, input):
        """This layer will receive an input tensor from 'input'."""
        self.inputs.append(input)
        input.outputs.append(self)

    def add_output(self, output):
        """"This layer will send an output tensor to 'output'."""
        self.outputs.append(output)
        output.inputs.append(self)

    def add_weights(self, w):
        self.weights.append(w)

def _create_layer_subclass(type_name, layer_field_name):
    """Generate a new Layer sub-class based on lbann.proto.

    'type_name' is the name of a message in lbann.proto,
    e.g. 'FullyConnected'. It will be the name of the generated
    sub-class.

    'layer_field_name' is the name of the corresponding field within
    the 'Layer' message in lbann.proto.

    """

    # Extract the names of all fields.
    layer_type = getattr(lbann_pb2, type_name)
    field_names = list(layer_type.DESCRIPTOR.fields_by_name.keys())

    # Sub-class constructor.
    def __init__(self, name='', inputs=[], outputs=[],
                 weights=[], data_layout='', **kwargs):
        super().__init__(self, name, inputs, outputs,
                         weights, data_layout)
        for field in kwargs:
            if field not in field_names:
                raise ValueError('Unknown argument {0}'.format(field))
        for field_name in field_names:
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            else:
                setattr(self, field_name, None)

    # Method for exporting a protobuf message.
    def export_proto(self):
        proto = super().export_proto()
        layer_message = getattr(proto, layer_field_name)
        layer_message.SetInParent() # Create empty message
        for field_name in field_names:
            v = getattr(self, field_name)
            if v is not None:
                setattr(layer_message, field_name, v)
        return proto

    # Create sub-class.
    return type(type_name, (Layer,),
                {'__init__': __init__, 'export_proto': export_proto})

# Generate Layer sub-classes from lbann.proto
# Note: The list of skip fields must be updated if any new fields are
# added to the Layer message in lbann.proto
_skip_fields = [
    'name', 'parents', 'children', 'data_layout', 'device_allocation',
    'weights', 'num_neurons_from_data_reader', 'freeze', 'hint_layer',
    'weights_data', 'top', 'bottom', 'type', 'motif_layer'
]
_generated_layers = {}
for field in lbann_pb2.Layer.DESCRIPTOR.fields:
    if field.name not in _skip_fields:
        type_name = field.message_type.name
        _generated_layers[type_name] = _create_layer_subclass(type_name, field.name)
_add_to_module_namespace(_generated_layers)

# Set up weight initializers.
class Initializer:
    """Base class for weight initializers."""

    def __init__(self):
        pass

    def proto(self, weights):
        """Add the initializer to the protobuf weights."""
        raise NotImplementedError('proto not implemented')

# Will hold all classes created by _create_proto_initializer.
_proto_initializers = {}
def _create_proto_initializer(type_name, message_name):
    """Create a new Initializer subclass for an initializer."""
    init_type = getattr(lbann_pb2, type_name)
    field_names = list(init_type.DESCRIPTOR.fields_by_name.keys())
    # Create init method which sets fields.
    def __init__(self, **kwargs):
        Initializer.__init__(self)
        for field in kwargs:
            if field not in field_names:
                raise ValueError('Unknown argument {0}'.format(field))
        for field_name in field_names:
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            else:
                setattr(self, field_name, None)
    # Method for adding to the protobuf weights.
    def proto(self, weights):
        init_field = getattr(weights, message_name)
        init_field.SetInParent()  # Create empty message.
        for field_name in field_names:
            v = getattr(self, field_name)
            if v is not None:
                setattr(init_field, field_name, v)
    _proto_initializers[type_name] = type(type_name, (Initializer,),
                                          {'__init__': __init__, 'proto': proto})

# Generate Initializer sub-classes.
for field in lbann_pb2.Weights.DESCRIPTOR.fields:
    if field.name not in ['name', 'optimizer']:
        _create_proto_initializer(field.message_type.name, field.name)
_add_to_module_namespace(_proto_initializers)

# Set up weights.
class Weights:
    """Base class for weights."""

    def __init__(self, name, initializer):
        """Initialize weights with name and an initializer."""
        self.name = name
        self.initializer = initializer

    def proto(self, model):
        """Add weights to the protobuf model."""
        weights = model.weights.add()
        weights.name = self.name
        self.initializer.proto(weights)

# Set up objective functions.
# Objective functions use only layer terms, and treats all other messages as
# regularization/etc. terms. This is not strictly true right now, but is the
# direction being moved in.
class ObjectiveRegularization:
    """Base class for objective function regularization terms."""

    def __init__(self):
        pass

    def proto(self, objfunc):
        """Add the regularization term to the protobuf objective function."""
        raise NotImplementedError('proto not implemented')

# Will hold all classes created by create_proto_objective_regularization.
_proto_objective_regularization = {}
def _create_proto_objective_regularization(type_name, message_name):
    """Create a new ObjectiveRegularization subclass for regularization."""
    objreg_type = getattr(lbann_pb2, type_name)
    field_names = list(objreg_type.DESCRIPTOR.fields_by_name.keys())
    # Create init method which sets fields.
    def __init__(self, **kwargs):
        ObjectiveRegularization.__init__(self)
        for field in kwargs:
            if field not in field_names:
                raise ValueError('Unknown argument {0}'.format(field))
        for field_name in field_names:
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            else:
                setattr(self, field_name, None)
    # Method for adding to the protobuf objective function.
    def proto(self, objfunc):
        objreg_field = getattr(objfunc, message_name)
        objreg = objreg_field.add()
        objreg.SetInParent()  # Needed to create empty messages.
        for field_name in field_names:
            v = getattr(self, field_name)
            if v is not None:
                # Ugly hack, special case for L2WeightRegularization:
                if field_name == 'weights':
                    setattr(objreg, field_name, ' '.join([w.name for w in v]))
                else:
                    setattr(objreg, field_name, v)
    _proto_objective_regularization[type_name] = type(
        type_name, (ObjectiveRegularization,),
        {'__init__': __init__, 'proto': proto})

# Generate ObjectiveRegularization sub-classes.
for field in lbann_pb2.ObjectiveFunction.DESCRIPTOR.fields:
    # Skip layer_term because we already use it, skip cross_entropy because it
    # is a real layer.
    # TODO: This should get cleaned up more, once we delete the old objective
    # functions from lbann.proto.
    if field.name not in ['layer_term', 'cross_entropy']:
        _create_proto_objective_regularization(field.message_type.name,
                                               field.name)
_add_to_module_namespace(_proto_objective_regularization)

class ObjectiveFunction:
    """Basic building block for objective functions."""

    def __init__(self, layers, regularization=[]):
        """Create an objective function with layer terms and regularization.

        layers is a layer, or a sequence of either layers or (layer, scale)
        tuples.

        """
        if type(layers) is not list:
            self.layers = [layers]
        else:
            self.layers = layers
        self.regularization = regularization

    def proto(self, model):
        """Add the objective function to the protobuf model."""
        objfunc = model.objective_function
        # Add layer terms.
        for layer in self.layers:
            term = objfunc.layer_term.add()
            if type(layer) is tuple:
                term.layer = layer[0].name
                term.scale_factor = layer[1]
            else:
                term.layer = layer.name
        # Add regularization terms.
        for objreg in self.regularization:
            objreg.proto(objfunc)

# Set up metrics.
# We only support layer metrics, so this is simple.
class Metric:
    """Basic building block for metrics."""

    def __init__(self, name, layer, unit):
        """Initialize a metric based of off layer."""
        self.name = name
        self.layer = layer
        self.unit = unit

    def proto(self, model):
        metric = model.metric.add()
        metric.layer_metric.name = self.name
        metric.layer_metric.layer = self.layer.name
        metric.layer_metric.unit = self.unit

# Set up callbacks.
class Callback:
    """Basic building block for callbacks."""

    def __init__(self): pass

    def proto(self, model):
        """Add this callback to the protobuf model."""
        raise NotImplementedError('proto not implemented')

    def _add_callback(self, model):
        """Add a callback message to model."""
        return model.callback.add()

_proto_callbacks = {}  # Will hold all classes created by create_proto_callback.
def _create_proto_callback(type_name, message_name):
    """Create a new Callback subclass for a callback."""
    callback_type = getattr(lbann_pb2, type_name)
    field_names = list(callback_type.DESCRIPTOR.fields_by_name.keys())
    # Create the init method which sets fields.
    def __init__(self, **kwargs):
        Callback.__init__(self)
        for field in kwargs:
            if field not in field_names:
                raise ValueError('Unknown argument {0}'.format(field))
        for field_name in field_names:
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            else:
                setattr(self, field_name, None)
    # Method for adding to the protobuf model.
    def proto(self, model):
        callback = self._add_callback(model)
        callback_field = getattr(callback, message_name)
        callback_field.SetInParent()  # Create empty message.
        for field_name in field_names:
            v = getattr(self, field_name)
            if v is not None:
                # Handle repeated fields.
                if type(v) is list:
                    field = getattr(callback_field, field_name)
                    field.extend(v)
                else:
                    setattr(callback_field, field_name, v)
    _proto_callbacks[type_name] = type(type_name, (Callback,),
                                       {'__init__': __init__, 'proto': proto})

# Generate Callback sub-classes.
for field in lbann_pb2.Callback.DESCRIPTOR.fields:
    _create_proto_callback(field.message_type.name, field.name)
_add_to_module_namespace(_proto_callbacks)

def traverse_layer_graph(start_layers):
    """Traverse a layer graph.

    The traversal starts from the entries in 'start_layers'. The
    traversal is in depth-first order, except that no layer is visited
    until all its inputs have been visited.

    """
    layers = []
    visited = set()
    stack = (start_layers
             if isinstance(start_layers, (list,))
             else [start_layers])
    while stack:
        l = stack.pop()
        layers.append(l)
        visited.add(l)
        for output in l.outputs:
            if (output not in visited):
                and all([(input in visited)
                         for input in output.inputs])):
                stack.append(output)
    return layers

def save_model(filename, layers, data_layout, mini_batch_size, epochs,
               objective_func, metrics=[], callbacks=[]):
    """Save a model to filename.

    Provide the first module (i.e. the input layer) to be saved.

    """
    pb = lbann_pb2.LbannPB()
    pb.model.data_layout = data_layout
    pb.model.mini_batch_size = mini_batch_size
    pb.model.block_size = 256  # TODO: Make configurable.
    pb.model.num_epochs = epochs
    pb.model.num_parallel_readers = 0  # TODO: Make configurable
    pb.model.procs_per_model = 0  # TODO: Make configurable
    # Add the objective function, metrics, and callbacks.
    objective_func.proto(pb.model)
    for metric in metrics:
        metric.proto(pb.model)
    for callback in callbacks:
        callback.proto(pb.model)
    layers = traverse_layer_graph(layers)
    pb.model.extend([l.export_proto() for l in layers])
    with open(filename, 'wb') as f:
        f.write(google.protobuf.text_format.MessageToString(
            pb, use_index_order=True).encode())
