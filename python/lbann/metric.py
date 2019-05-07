"""Neural network tensor operations."""
import abc
from lbann import lbann_pb2

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
