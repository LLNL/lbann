"""LBANN Trainer."""
import abc
from lbann import lbann_pb2
from lbann.util import make_iterable

class Trainer:
    """LBANN Trainer."""

    def __init__(self):
        # Scalar fields
        self.block_size = 256           # TODO: Make configurable
        self.procs_per_trainer = 0      # TODO: Make configurable

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Initialize protobuf message
        trainer = lbann_pb2.Trainer()
        trainer.block_size = self.block_size
        trainer.procs_per_trainer = self.procs_per_trainer

        return trainer

    def save_proto(self, filename):
        """Export trainer to prototext file."""
        save_prototext(filename, trainer=self.export_proto())
