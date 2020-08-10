"""Neural network model."""
from lbann import model_pb2
from lbann.util import make_iterable
import lbann.core.layer
import lbann.core.objective_function

class Model:
    """Neural network model."""

    def __init__(self, epochs,vector_communication=0,
                 layers=[], weights=[], objective_function=None,
                 metrics=[], callbacks=[],
                 summary_dir=None,serialize_io=False):

        # Scalar fields
        self.epochs = epochs
        self.summary_dir = summary_dir
        self.serialize_io = serialize_io
        # Get connected layers
        self.layers = list(lbann.core.layer.traverse_layer_graph(layers))

        # Get weights associated with layers
        self.weights = set(make_iterable(weights))
        for l in self.layers:
            self.weights.update(l.weights)

        # Construct objective function if needed
        obj_type = lbann.core.objective_function.ObjectiveFunction
        if isinstance(objective_function, obj_type):
            self.objective_function = objective_function
        elif objective_function is None:
            self.objective_function = obj_type()
        else:
            self.objective_function = obj_type(objective_function)

        # Metrics and callbacks
        self.metrics = make_iterable(metrics)
        self.callbacks = make_iterable(callbacks)
        self.vector_communication = vector_communication

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Initialize protobuf message
        model = model_pb2.Model()
        model.num_epochs = self.epochs
        model.vector_communication = self.vector_communication
        if self.summary_dir is not None:
            model.summarizer.dir = self.summary_dir
        model.serialize_io = self.serialize_io
        # Add model components
        model.layer.extend([l.export_proto() for l in self.layers])
        model.weights.extend([w.export_proto() for w in self.weights])
        model.objective_function.CopyFrom(self.objective_function.export_proto())
        model.metric.extend([m.export_proto() for m in self.metrics])
        model.callback.extend([c.export_proto() for c in self.callbacks])

        return model
