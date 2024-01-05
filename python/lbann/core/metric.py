"""Neural network tensor operations."""
from lbann import metrics_pb2
from typing import List, Union


class BaseMetric:
    pass


class Metric(BaseMetric):
    """Metric that takes value from a layer.

    Corresponds to a "layer metric" in LBANN. This may need to be
    generalized if any other LBANN metrics are implemented.

    """

    def __init__(self, layer, name=None, unit=''):
        """Initialize a metric based off of a layer."""
        self.layer = layer
        self.name = name if name else self.layer.name
        self.unit = unit

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = metrics_pb2.Metric()
        proto.layer_metric.layer = self.layer.name
        proto.layer_metric.name = self.name
        proto.layer_metric.unit = self.unit
        return proto
