"""LBANN Trainer."""
from lbann import trainer_pb2
from lbann.util import make_iterable

class Trainer:
    """Manages the training of a neural network model."""

    def __init__(self,
                 mini_batch_size,
                 name=None,
                 procs_per_trainer=None,
                 num_parallel_readers=None,
                 random_seed=None,
                 callbacks=[]):
        self.name = name
        self.procs_per_trainer = procs_per_trainer
        self.num_parallel_readers = num_parallel_readers
        self.random_seed = random_seed
        self.mini_batch_size = mini_batch_size
        self.hydrogen_block_size = None
        # Callbacks
        self.callbacks = make_iterable(callbacks)

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
        if self.random_seed is not None:
            trainer.random_seed = self.random_seed
        if self.mini_batch_size is not None:
            trainer.mini_batch_size = self.mini_batch_size
        if self.hydrogen_block_size is not None:
            trainer.hydrogen_block_size = self.hydrogen_block_size

        # Add trainer components
        trainer.callback.extend([c.export_proto() for c in self.callbacks])

        return trainer
