"""Neural network tensor operations."""
import abc
from lbann import lbann_pb2
from lbann.util import make_iterable
import lbann.util.class_generator

class Layer(abc.ABC):
    """Neural network tensor operation."""

    global_count = 0  # Static counter, used for default names

    def __init__(self,
                 parents=[],
                 children=[],
                 weights=[],
                 name=None,
                 device=None,
                 data_layout=None,
                 hint_layer=None):
        """Constructor.

        Args:
            parents (Iterable of Layer, optional): Sources of input
                tensors.
            children (Iterable of Layer, optional): Destinations of
                output tensors.
            weights (Iterable of Weights, optional): Trainable
                parameters.
            name (str, optional): Unique identifier (default is
                'layer<index>').
            device (str, optional): Device to use, e.g. CPU or GPU.
            data_layout (str, optional): Data distribution scheme.
            hint_layer (Layer, optional): Hint for output dimensions.

        """
        Layer.global_count += 1
        self.parents = []
        self.children = []
        self.weights = []
        self.name = name if name else 'layer{0}'.format(Layer.global_count)
        self.device = device
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
        if self.device:
            proto.device_allocation = self.device
        if self.data_layout:
            proto.data_layout = self.data_layout
        if self.hint_layer:
            proto.hint_layer = self.hint_layer.name
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
        'name', 'device', 'data_layout', 'hint_layer']),
    base_has_export_proto = True)
for c in classes:
    globals()[c.__name__] = c

def traverse_layer_graph(layers):
    """Topologically ordered traversal of layer graph.

    All layers that are connected to `layers` will be traversed. The
    layer graph is assumed to be acyclic. No checks are made for
    cycles and strange things may happen if one exists.

    Args:
        layers (Layer or Iterator of Layer): Node(s) in layer graph.

    Yields:
        Layer: Node in layer graph, in a topological order.

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
