"""Trainable model parameters."""
import abc
from lbann import weights_pb2
import lbann.util.class_generator

class Initializer(abc.ABC):
    """Initialization scheme for `Weights`."""
    def export_proto(self):
        """Construct and return a protobuf message."""
        return weights_pb2.Initializer()

# Generate Initializer sub-classes from weights.proto.
classes = lbann.util.class_generator.generate_classes_from_protobuf_message(
    weights_pb2.Initializer,
    base_class = Initializer,
    base_has_export_proto = True)
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
        proto = weights_pb2.Weights()
        proto.name = self.name

        # Set initializer if needed
        if self.initializer:
            proto.initializer.CopyFrom(self.initializer.export_proto())
            proto.initializer.SetInParent()

        # Set optimizer if needed
        if self.optimizer:
            proto.optimizer.CopyFrom(self.optimizer.export_proto())
            proto.optimizer.SetInParent()

        return proto
