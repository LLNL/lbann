"""Neural network modules for graph convolutional models.""" 
from .GINConv  import GINConv
from .GCNConv import GCNConv
__all__ = [
    'GCNConv',
    'GINConv', 
    ]
