"""Trainable model parameters."""
import abc
from lbann import weights_pb2
from enum import Enum
from typing import Optional
import lbann.core.util

class Initializer(abc.ABC):
    """Initialization scheme for `Weights`.

        datatype (lbann.DataType, optional): Data type used for weights.

    """
    def export_proto(self):
        """Construct and return a protobuf message."""
        return weights_pb2.Initializer()

# Generate Initializer sub-classes from weights.proto.
if weights_pb2:
    classes = lbann.core.util.generate_classes_from_protobuf_message(
        weights_pb2.Initializer,
        base_class = Initializer,
        base_has_export_proto = True)
    for c in classes:
        globals()[c.__name__] = c


class ShardingStrategy(Enum):
        FULL = 0      # Sharded across all ranks (STAR x VC)
        GRID_ROWS = 1 # Sharded across the process grid rows (STAR x MC)
        GRID_COLS = 2 # Sharded across the process grid columns (STAR x MR)


class Weights:
    """Trainable parameters for neural network."""

    global_count = 0  # Static counter, used for default names

    def __init__(self, initializer=None, optimizer=None, name=None,
                 datatype=None, sharded=None,
                 sharding_strategy: Optional[ShardingStrategy] = None):
        Weights.global_count += 1
        self.name = name if name else 'weights{0}'.format(Weights.global_count)
        self.initializer = initializer
        self.optimizer = optimizer
        self.datatype = datatype
        self.sharded = sharded
        self.sharding_strategy = sharding_strategy

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

        # Set datatype if needed
        if self.datatype:
            proto.datatype = self.datatype

        if self.sharded:
            proto.sharded = self.sharded

        if self.sharding_strategy is not None:
            proto.sharding_strategy = self.sharding_strategy.value

        return proto
