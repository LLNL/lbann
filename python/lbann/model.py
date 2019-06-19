"""Neural network model."""
import abc
from lbann import lbann_pb2
from lbann.util import make_iterable
import lbann.layer
import lbann.objective_function

class Model:
    """Neural network model."""

    def __init__(self, mini_batch_size, epochs,
                 layers=[], weights=[], objective_function=None,
                 metrics=[], callbacks=[], random_seed=None):

        # Scalar fields
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.block_size = 256           # TODO: Make configurable
        self.num_parallel_readers = 0   # TODO: Make configurable
        self.procs_per_trainer = 0      # TODO: Make configurable
        self.random_seed = random_seed

        # Get connected layers
        self.layers = list(lbann.layer.traverse_layer_graph(layers))

        # Get weights associated with layers
        self.weights = set(make_iterable(weights))
        for l in self.layers:
            self.weights.update(l.weights)

        # Construct objective function if needed
        obj_type = lbann.objective_function.ObjectiveFunction
        if isinstance(objective_function, obj_type):
            self.objective_function = objective_function
        elif objective_function is None:
            self.objective_function = obj_type()
        else:
            self.objective_function = obj_type(objective_function)

        # Metrics and callbacks
        self.metrics = make_iterable(metrics)
        self.callbacks = make_iterable(callbacks)

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Initialize protobuf message
        model = lbann_pb2.Model()
        model.mini_batch_size = self.mini_batch_size
        model.num_epochs = self.epochs
        model.block_size = self.block_size
        model.num_parallel_readers = self.num_parallel_readers
        model.procs_per_trainer = self.procs_per_trainer
        if self.random_seed is not None:
            model.random_seed = self.random_seed

        # Add model components
        model.layer.extend([l.export_proto() for l in self.layers])
        model.weights.extend([w.export_proto() for w in self.weights])
        model.objective_function.CopyFrom(self.objective_function.export_proto())
        model.metric.extend([m.export_proto() for m in self.metrics])
        model.callback.extend([c.export_proto() for c in self.callbacks])

        return model

    def save_proto(self, filename):
        """Export model to prototext file."""
        save_prototext(filename, model=self.export_proto())
