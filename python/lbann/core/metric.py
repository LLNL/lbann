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


class ExecutableMetric(BaseMetric):
    """Metric that takes its value from a printout of a binary executable.
    """

    def __init__(self,
                 name: str,
                 filename: str,
                 other_args: Union[str, List[str]] = ''):
        self.name = name
        self.filename = filename
        if isinstance(other_args, str):
            self.other_args = other_args
        else:
            self.other_args = " ".join(other_args)

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = metrics_pb2.Metric()
        proto.executable_metric.name = self.name
        proto.executable_metric.filename = self.filename
        proto.executable_metric.other_args = self.other_args
        return proto


class PythonMetric(BaseMetric):
    """Metric that takes its value from a result of a Python function.
    """

    def __init__(self,
                 name: str,
                 module: str,
                 module_dir: str,
                 function: str = 'evaluate'):
        self.name = name
        self.module = module
        self.module_dir = module_dir
        self.function = function

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = metrics_pb2.Metric()
        proto.python_metric.name = self.name
        proto.python_metric.module = self.module
        proto.python_metric.module_dir = self.module_dir
        proto.python_metric.function = self.function
        return proto
