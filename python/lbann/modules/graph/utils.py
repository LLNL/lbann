import lbann
from lbann.util import str_list


def GraphExpand(features, indices, name=None):
    """Places the features according the indices to an expanded matrix

       output[i] = features[indices[i]]

       Args:
            features (Layer) : 2D matrix with shape (N, F)
            indices (Layer): 1D matrix with shape (E)
       returnL (Layer) of shape (E,F)
    """
    GraphExpand.count += 1
    if (name is None):
        name = f"graph_expand_{GraphExpand.count}" 
    return lbann.Gather(features, indices, axis=0, name=name)

def GraphReduce(features, indices, dims, name=None):
    """Performs a sum-reduction of the features according the indices.
       output[indices[i]] += features[i] 
        
       Args:
            features (layer) : 2D matrix with shape (E, F)
            indices (layer): 1D matrix with shape (E)
            dims (list of int): tuple of ints with the values (N, F)
       returns: (layer) of shape (N, F)
    """
    GraphReduce.count += 1
    if (name is None):
        name = f"graph_reduce_{GraphReduce.count}"
    return lbann.Scatter(features, indices, dims=str_list(dims), axis=0, name=name)

GraphReduce.count = 0
GraphExpand.count = 0
