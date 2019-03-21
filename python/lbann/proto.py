"""Generate prototexts for LBANN models."""

import google.protobuf.text_format
import google.protobuf.message
from lbann import lbann_pb2
from lbann.util import make_iterable
import lbann.util.class_generator

# ==============================================
# Layers
# ==============================================

class Layer:
    """Base class for layers."""

    global_count = 0  # Static counter, used for default names

    def __init__(self, parents = [], children = [], weights = [],
                 name = None, data_layout = 'data_parallel',
                 hint_layer = None):
        Layer.global_count += 1
        self.parents = []
        self.children = []
        self.weights = []
        self.name = name if name else 'layer{0}'.format(Layer.global_count)
        self.data_layout = data_layout
        self.hint_layer = hint_layer

        # Initialize parents, children, and weights
        for l in make_iterable(parents):
            self.add_parent(l)
        for l in make_iterable(children):
            self.add_child(child)
        for w in make_iterable(weights):
            self.add_weights(w)

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.Layer()
        proto.parents = ' '.join([l.name for l in self.parents])
        proto.children = ' '.join([l.name for l in self.children])
        proto.weights = ' '.join([w.name for w in self.weights])
        proto.name = self.name
        proto.data_layout = self.data_layout
        proto.hint_layer = self.hint_layer.name if self.hint_layer else ''
        return proto

    def add_parent(self, parent):
        """This layer will receive an input tensor from `parent`."""
        for p in make_iterable(parent):
            self.parents.append(p)
            p.children.append(self)

    def add_child(self, child):
        """"This layer will send an output tensor to `child`."""
        for c in make_iterable(child):
            self.children.append(c)
            c.parents.append(self)

    def add_weights(self, w):
        """Add w to this layer's weights."""
        self.weights.extend(make_iterable(w))

    def __call__(self, parent):
        """This layer will recieve an input tensor from `parent`.

        Syntactic sugar around `add_parent` function.

        """
        self.add_parent(parent)

# Generate Layer sub-classes from lbann.proto
# Note: The list of skip fields must be updated if any new fields are
# added to the Layer message in lbann.proto
classes = lbann.util.class_generator.generate_classes_from_protobuf_message(
    lbann_pb2.Layer,
    skip_fields = set([
        'name', 'parents', 'children', 'data_layout', 'device_allocation',
        'weights', 'num_neurons_from_data_reader', 'freeze', 'hint_layer',
        'weights_data', 'top', 'bottom', 'type', 'motif_layer']),
    base_class = Layer,
    base_kwargs = set([
        'parents', 'children', 'weights',
        'name', 'data_layout', 'hint_layer']),
    base_has_export_proto = True)
for c in classes:
    globals()[c.__name__] = c

def traverse_layer_graph(layers):
    """Generator function for a topologically ordered graph traversal.

    `layers` should be a `Layer` or a sequence of `Layer`s. All layers
    that are connected to `layers` will be traversed.

    The layer graph is assumed to be acyclic. Strange things may
    happen if this does not hold.

    """

    # DFS to find root nodes in layer graph
    roots = []
    visited = set()
    stack = list(make_iterable(layers))
    while stack:
        l = stack.pop()
        if l not in visited:
            visited.add(l)
            stack.extend(l.parents)
            stack.extend(l.children)
            if not l.parents:
                roots.append(l)

    # DFS to traverse layer graph in topological order
    visited = set()
    stack = roots
    while stack:
        l = stack.pop()
        if (l not in visited
            and all([(p in visited) for p in l.parents])):
            visited.add(l)
            stack.extend(l.children)
            yield l

# ==============================================
# Weights and weight initializers
# ==============================================

# Set up weight initializers.
class Initializer:
    """Base class for weight initializers."""

    def __init__(self):
        pass

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Should be overridden in all sub-classes
        raise NotImplementedError

# Generate Initializer sub-classes from lbann.proto.
# Note: The list of skip fields must be updated if any new fields are
# added to the Weights message in lbann.proto
classes = lbann.util.class_generator.generate_classes_from_protobuf_message(
    lbann_pb2.Weights,
    skip_fields = set(['name', 'optimizer']),
    base_class = Initializer)
for c in classes:
    globals()[c.__name__] = c

class Weights:
    """Trainable model parameters."""

    global_count = 0  # Static counter, used for default names

    def __init__(self, initializer=None, optimizer=None, name=None):
        Weights.global_count += 1
        self.name = name if name else 'weights{0}'.format(Weights.global_count)
        self.initializer = initializer
        self.optimizer = optimizer

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.Weights()
        proto.name = self.name

        # Set initializer if needed
        if self.initializer:
            type_name = type(self.initializer).__name__
            field_name = None
            for field in lbann_pb2.Weights.DESCRIPTOR.fields:
                if field.message_type and field.message_type.name == type_name:
                    field_name = field.name
                    break
            init_message = getattr(proto, field_name)
            init_message.CopyFrom(self.initializer.export_proto())
            init_message.SetInParent()

        # Set optimizer if needed
        if self.optimizer:
            proto.optimizer.CopyFrom(self.optimizer.export_proto())
            proto.optimizer.SetInParent()

        return proto

# ==============================================
# Objective functions
# ==============================================

# Note: Currently, only layer terms and L2 weight regularization terms
# are supported in LBANN. If more terms are added, it may be
# worthwhile to autogenerate sub-classes of ObjectiveFunctionTerm.

class ObjectiveFunctionTerm:
    """Base class for objective function terms."""

    def __init__(self):
        pass

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Should be overridden in all sub-classes
        raise NotImplementedError

class LayerTerm(ObjectiveFunctionTerm):
    """Objective function term that takes value from a layer."""

    def __init__(self, layer, scale=1.0):
        self.layer = layer
        self.scale = scale

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.LayerTerm()
        proto.layer = self.layer.name
        proto.scale_factor = self.scale
        return proto

class L2WeightRegularization(ObjectiveFunctionTerm):
    """Objective function term for L2 regularization on weights."""

    def __init__(self, weights=[], scale=1.0):
        self.scale = scale
        self.weights = list(make_iterable(weights))

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.L2WeightRegularization()
        proto.scale_factor = self.scale
        proto.weights = ' '.join([w.name for w in self.weights])
        return proto

class ObjectiveFunction:
    """Objective function for optimization algorithm."""

    def __init__(self, terms=[]):
        """Create an objective function with layer terms and regularization.

        `terms` should be a sequence of `ObjectiveFunctionTerm`s and
        `Layer`s.

        """
        self.terms = []
        for t in make_iterable(terms):
            self.add_term(t)

    def add_term(self, term):
        """Add a term to the objective function.

        `term` may be a `Layer`, in which case a `LayerTerm` is
        constructed and added to the objective function.

        """
        if isinstance(term, Layer):
            term = LayerTerm(term)
        self.terms.append(term)

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.ObjectiveFunction()
        for term in self.terms:
            term_message = term.export_proto()
            if type(term) is LayerTerm:
                proto.layer_term.extend([term_message])
            elif type(term) is L2WeightRegularization:
                proto.l2_weight_regularization.extend([term_message])
        return proto

# ==============================================
# Metrics
# ==============================================

class Metric:
    """Metric that takes value from a layer.

    Corresponds to a "layer metric" in LBANN. This may need to be
    generalized if any other LBANN metrics are implemented.

    """

    def __init__(self, layer, name=None, unit=''):
        """Initialize a metric based of off layer."""
        self.layer = layer
        self.name = name if name else self.layer.name
        self.unit = unit

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.Metric()
        proto.layer_metric.layer = self.layer.name
        proto.layer_metric.name = self.name
        proto.layer_metric.unit = self.unit
        return proto

# ==============================================
# Callbacks
# ==============================================

class Callback:
    """Base class for callbacks."""

    def __init__(self):
        pass

    def export_proto(self):
        """Construct and return a protobuf message."""
        return lbann_pb2.Callback()

# Generate Callback sub-classes from lbann.proto
# Note: The list of skip fields must be updated if any new fields are
# added to the Callback message in lbann.proto
classes = lbann.util.class_generator.generate_classes_from_protobuf_message(
    lbann_pb2.Callback,
    base_class = Callback,
    base_has_export_proto = True)
for c in classes:
    globals()[c.__name__] = c

# ==============================================
# Optimizers
# ==============================================

class Optimizer:
    """Base class for optimizers."""

    def __init__(self):
        pass

    def export_proto(self):
        """Construct and return a protobuf message."""
        return lbann_pb2.Optimizer()

# Generate Optimizer sub-classes from lbann.proto
# Note: The list of skip fields must be updated if any new fields are
# added to the Optimizer message in lbann.proto
classes = lbann.util.class_generator.generate_classes_from_protobuf_message(
    lbann_pb2.Optimizer,
    base_class = Optimizer,
    base_has_export_proto = True)
for c in classes:
    globals()[c.__name__] = c

# ==============================================
# Model
# ==============================================

class Model:
    """Neural network model."""

    def __init__(self, mini_batch_size, epochs,
                 layers=[], weights=[], objective_function=None,
                 metrics=[], callbacks=[]):

        # Scalar fields
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.block_size = 256           # TODO: Make configurable
        self.num_parallel_readers = 0   # TODO: Make configurable
        self.procs_per_trainer = 0      # TODO: Make configurable

        # Get connected layers
        self.layers = list(traverse_layer_graph(layers))

        # Get weights associated with layers
        self.weights = set(make_iterable(weights))
        for l in self.layers:
            self.weights.update(l.weights)

        # Construct objective function if needed
        if isinstance(objective_function, ObjectiveFunction):
            self.objective_function = objective_function
        elif objective_function is None:
            self.objective_function = ObjectiveFunction()
        else:
            self.objective_function = ObjectiveFunction(objective_function)

        # Metrics and callbacks
        self.metrics = make_iterable(metrics)
        self.callbacks = make_iterable(callbacks)

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Initialize protobuf message
        model = lbann_pb2.Model()
        model.mini_batch_size = self.mini_batch_size
        model.num_epochs = self.epochs
        model.block_size = self.block_size
        model.num_parallel_readers = self.num_parallel_readers
        model.procs_per_trainer = self.procs_per_trainer

        # Add model components
        model.layer.extend([l.export_proto() for l in self.layers])
        model.weights.extend([w.export_proto() for w in self.weights])
        model.objective_function.CopyFrom(self.objective_function.export_proto())
        model.metric.extend([m.export_proto() for m in self.metrics])
        model.callback.extend([c.export_proto() for c in self.callbacks])

        return model

    def save_proto(self, filename):
        """Export model to prototext file."""
        save_prototext(filename, model=self.export_proto())

# ==============================================
# Export models
# ==============================================

def save_prototext(filename, **kwargs):
    """Save a prototext.
    This function accepts the LbannPB objects via `kwargs`, such as
    `model`, `data_reader`, and `optimizer`.
    """

    # Construct protobuf message
    for key, value in kwargs.items():
        if not isinstance(value, google.protobuf.message.Message):
            kwargs[key] = value.export_proto()
    pb = lbann_pb2.LbannPB(**kwargs)

    # Write to file
    with open(filename, 'wb') as f:
        f.write(google.protobuf.text_format.MessageToString(
            pb, use_index_order=True).encode())
