import abc
from lbann import optimizers_pb2
import lbann.core.util

class Optimizer(abc.ABC):
    """Optimization algorithm for a neural network's parameters."""
    def export_proto(self):
        """Construct and return a protobuf message."""
        return optimizers_pb2.Optimizer()

# Generate Optimizer sub-classes from lbann.proto
# Note: The list of skip fields must be updated if any new fields are
# added to the Optimizer message in lbann.proto
classes = lbann.core.util.generate_classes_from_protobuf_message(
    optimizers_pb2.Optimizer,
    base_class = Optimizer,
    base_has_export_proto = True)
for c in classes:
    globals()[c.__name__] = c
