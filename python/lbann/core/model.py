"""Neural network model."""

from typing import NamedTuple, Optional

from lbann import model_pb2
from lbann.util import make_iterable
import lbann.core.layer
import lbann.core.objective_function
from enum import Enum

class SubgraphCommunication(Enum):
    PT2PT = 0
    COLL = 1
    COLL_OPT = 2

def convert_to_protbuf_enums(subgraph_communication):
    if(subgraph_communication==SubgraphCommunication.PT2PT):
        return model_pb2.SubGraphCommunication.Value('PT2PT')
    elif (subgraph_communication==SubgraphCommunication.COLL):
        return model_pb2.SubGraphCommunication.Value('COLL')
    elif (subgraph_communication==SubgraphCommunication.COLL_OPT):
        return model_pb2.SubGraphCommunication.Value('COLL_OPT')


class AmpOptions(NamedTuple):
    """Options for automatic mixed precision."""
    enabled: bool = False
    init_scale: Optional[float] = None
    growth_factor: Optional[float] = None
    backoff_factor: Optional[float] = None
    growth_interval: Optional[int] = None


class Model:
    """Neural network model."""

    def __init__(self, epochs,
                 layers=[], weights=[], objective_function=None,
                 metrics=[], callbacks=[],
                 name=None,
                 summary_dir=None,
                 subgraph_communication=SubgraphCommunication.PT2PT,
                 subgraph_topology=False,
                 subgraph_num_common_resources=0,
                 amp: AmpOptions = None):

        # Scalar fields
        self.epochs = epochs
        self.name = name
        self.summary_dir = summary_dir

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
        self.subgraph_communication = subgraph_communication
        self.subgraph_topology = subgraph_topology
        self.subgraph_num_common_resources = subgraph_num_common_resources

        # AMP.
        self.amp = amp

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Initialize protobuf message
        model = model_pb2.Model()
        if self.name is not None:
            model.name = self.name
        model.num_epochs = self.epochs
        model.subgraph_communication = convert_to_protbuf_enums(self.subgraph_communication)
        model.enable_subgraph_topology = self.subgraph_topology
        model.subgraph_parent_grid_resources = self.subgraph_num_common_resources
        if self.summary_dir is not None:
            model.summarizer.dir = self.summary_dir
        # Add model components
        model.layer.extend([l.export_proto() for l in self.layers])
        model.weights.extend([w.export_proto() for w in self.weights])
        model.objective_function.CopyFrom(self.objective_function.export_proto())
        model.metric.extend([m.export_proto() for m in self.metrics])
        model.callback.extend([c.export_proto() for c in self.callbacks])

        # Add AMP options:
        if self.amp is not None:
            model.amp.enabled = self.amp.enabled
            if self.amp.init_scale is not None:
                model.amp.init_scale = self.amp.init_scale
            if self.amp.growth_factor is not None:
                model.amp.growth_factor = self.amp.growth_factor
            if self.amp.backoff_factor is not None:
                model.amp.backoff_factor = self.amp.backoff_factor
            if self.amp.growth_interval is not None:
                model.amp.growth_interval = self.amp.growth_interval

        return model

    def __call__(self, *args, **kwargs):
        from lbann.core.evaluate import evaluate  # Avoid circular imports
        return evaluate(self, *args, **kwargs)
