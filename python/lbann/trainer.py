"""LBANN Trainer."""
import abc
from lbann import trainer_pb2
from lbann.util import make_iterable

class Trainer:
    """LBANN Trainer."""

    def __init__(self):
        # Scalar fields
        self.block_size = 256           # TODO: Make configurable
        self.procs_per_trainer = 0      # TODO: Make configurable
        self.num_parallel_readers = 0   # TODO: Make configurable
        self.num_gpus = 1               # TODO: Make configurable

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Initialize protobuf message
        trainer = trainer_pb2.Trainer()
        trainer.block_size = self.block_size
        trainer.procs_per_trainer = self.procs_per_trainer
        trainer.num_parallel_readers = self.num_parallel_readers
        trainer.num_gpus = self.num_gpus

        return trainer
