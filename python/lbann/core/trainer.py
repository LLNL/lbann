"""LBANN Trainer."""
from lbann import trainer_pb2
from lbann.util import make_iterable

class Trainer:
    """Manages the training of a neural network model."""

    def __init__(self,
                 mini_batch_size,
                 name=None,
                 random_seed=None,
                 serialize_io=None,
                 training_algo=None,
                 callbacks=[]):
        self.name = name
        self.random_seed = random_seed
        self.serialize_io = serialize_io
        self.mini_batch_size = mini_batch_size
        self.hydrogen_block_size = None
        self.training_algo = training_algo
        # Callbacks
        self.callbacks = make_iterable(callbacks)

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Initialize protobuf message
        trainer = trainer_pb2.Trainer()
        if self.name is not None:
            trainer.name = self.name
        if self.random_seed is not None:
            trainer.random_seed = self.random_seed
        if self.mini_batch_size is not None:
            trainer.mini_batch_size = self.mini_batch_size
        if self.hydrogen_block_size is not None:
            trainer.hydrogen_block_size = self.hydrogen_block_size
        if self.serialize_io is not None:
            trainer.serialize_io = self.serialize_io
        if self.training_algo is not None:
            trainer.training_algorithm.CopyFrom(self.training_algo.export_proto())

        # Add trainer components
        trainer.callback.extend([c.export_proto() for c in self.callbacks])

        return trainer
