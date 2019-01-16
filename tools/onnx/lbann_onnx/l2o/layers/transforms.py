import lbann_onnx
from lbann_onnx.util import getNodeAttributeByName
from lbann_onnx.l2o.util import parseSpatialAttributes
import onnx
import numpy as np

def parse_pooling(lp, inputShapes):
    return {"op": {"max": "MaxPool",
                   "average": "AveragePool"}[lp.pool_mode],
            "attrs": parseSpatialAttributes(lp, "pool", False)}

def parse_unpooling(lp, inputShapes):
    # lp is an ONNX Pooling node only in this parser function
    return {"op": "MaxUnpool",
            "attrs": dict(map(lambda x: (x, getNodeAttributeByName(lp, x).ints),
                              ["kernel_shape", "pads", "strides"]))}

def parse_slice(lp, inputShapes):
    offsets = list(map(int, lp.slice_points.split(" ")))
    sizes = list(map(lambda x: offsets[x+1]-offsets[x], range(len(offsets)-1)))
    return {"op": "Split",
            "attrs": {"axis": lp.slice_axis,
                      "split": sizes}}

def parse_concatenation(lp, inputShapes):
    return {"op": "Concat",
            "attrs": {"axis": lp.concatenation_axis}}

def parse_gaussian(lp, inputShape):
    # mean, stdev, neuron_dims
    return {"op": "RandomNormal",
            "outputCount": 1,
            "attrs": {"dtype": lbann_onnx.ELEM_TYPE,
                      "mean": lp.mean,
                      "scale": lp.stdev,
                      "shape": lp.neuron_dims if isinstance(lp.neuron_dims, list) \
                      else list(map(int, lp.neuron_dims.split(" ")))}}

def parse_reshape(lp, inputShape):
    shape = list(map(int, lp.dims.split(" ")))
    return {"op": "Reshape",
            "paramCount": 1,
            "inits": [{"shape": np.array(shape).shape,
                       "dataType": onnx.TensorProto.INT64,
                       "value": np.array(shape, dtype=np.int64).tobytes()}]}

def parse_reduction(lp, inputShapes):
    return {"op": {"sum": "ReduceSum",
                   "average": "ReduceMean"}[lp.mode],
            "attrs": {"keepdims": 0}}

##
## Dummy parsers
##

def parse_evaluation(lp, inputShapes):
    return {"op": "LbannEvaluation"}

def parse_zero(lp, inputShape):
    return {"op": "Identity"} # TODO: this is a dummy operation to perform correct infer_shape
