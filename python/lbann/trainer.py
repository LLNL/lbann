"""LBANN Trainer."""
import abc
from lbann import trainer_pb2
from lbann.util import make_iterable

class Trainer:
    """LBANN Trainer."""

    def __init__(self,
                 name=None,
                 procs_per_trainer=None,
                 num_parallel_readers=None):
        self.name = name
        self.procs_per_trainer = procs_per_trainer
        self.num_parallel_readers = num_parallel_readers
        self.hydrogen_block_size = None

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Initialize protobuf message
        trainer = trainer_pb2.Trainer()
        if self.name is not None:
            trainer.name = self.name
        if self.procs_per_trainer is not None:
            trainer.procs_per_trainer = self.procs_per_trainer
        if self.num_parallel_readers is not None:
            trainer.num_parallel_readers = self.num_parallel_readers
        if self.hydrogen_block_size is not None:
            trainer.hydrogen_block_size = self.hydrogen_block_size
        return trainer
