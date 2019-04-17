"""Trainable model parameters."""
import abc
from lbann import lbann_pb2
import lbann.util.class_generator

class Initializer(abc.ABC):
    """Initialization scheme for `Weights`."""
    def export_proto(self):
        pass

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
