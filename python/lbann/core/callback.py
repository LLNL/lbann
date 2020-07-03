"""Callbacks for neural network training."""
import abc
from lbann import callbacks_pb2
import lbann.core.util

class Callback(abc.ABC):
    """Callback for neural network training."""

    def __init__(self):
        pass

    def export_proto(self):
        """Construct and return a protobuf message."""
        return callbacks_pb2.Callback()

# Generate Callback sub-classes from lbann.proto
# Note: The list of skip fields must be updated if any new fields are
# added to the Callback message in lbann.proto
classes = lbann.core.util.generate_classes_from_protobuf_message(
    callbacks_pb2.Callback,
    base_class = Callback,
    base_has_export_proto = True)
for c in classes:
    globals()[c.__name__] = c

class ImageSelectionStrategy(abc.ABC):
    """Image selection strategy for summarize images callback."""

    def __init__(self):
        pass

    def export_proto(self):
        """Construct and return a protobuf message."""
        return callbacks_pb2.Callback.CallbackSummarizeImages.SelectionStrategy()

# Build all subclasses
classes = lbann.core.util.generate_classes_from_protobuf_message(
    callbacks_pb2.Callback.CallbackSummarizeImages.SelectionStrategy,
    base_class = ImageSelectionStrategy,
    base_has_export_proto = True)
for c in classes:
    globals()[c.__name__] = c
