import abc
from lbann import lbann_pb2
from lbann.util import make_iterable
import lbann.layer

# Note: Currently, only layer terms and L2 weight regularization terms
# are supported in LBANN. If more terms are added, it may be
# worthwhile to autogenerate sub-classes of ObjectiveFunctionTerm.

class ObjectiveFunctionTerm(abc.ABC):
    def export_proto(self): pass

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
        if isinstance(term, lbann.layer.Layer):
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
