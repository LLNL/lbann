import lbann_onnx
from lbann_onnx.util import getNodeAttributeByName
from lbann_onnx.l2o.util import parseSpatialAttributes
from lbann_onnx.l2o.layers import LbannLayerParser
import onnx
import numpy as np

class LbannLayerParser_pooling(LbannLayerParser):
    def parse(self):
        params = self.l.pooling
        return {"op": {"max": "MaxPool",
                       "average": "AveragePool"}[params.pool_mode],
                "attrs": parseSpatialAttributes(params, "pool", False)}

class LbannLayerParser_unpooling(LbannLayerParser):
    def parse(self):
        # OPTIMIZE: self.l is an ONNX Pooling node only in this parser function
        return {"op": "MaxUnpool",
                "attrs": dict(map(lambda x: (x, getNodeAttributeByName(self.l, x).ints),
                                  ["kernel_shape", "pads", "strides"]))}

class LbannLayerParser_slice(LbannLayerParser):
    def parse(self):
        params = self.l.slice
        offsets = list(map(int, params.slice_points.split(" ")))
        sizes = list(map(lambda x: offsets[x+1]-offsets[x], range(len(offsets)-1)))
        return {"op": "Split",
                "attrs": {"axis": params.slice_axis,
                          "split": sizes}}

class LbannLayerParser_concatenation(LbannLayerParser):
    def parse(self):
        return {"op": "Concat",
                "attrs": {"axis": self.l.concatenation.concatenation_axis}}

class LbannLayerParser_gaussian(LbannLayerParser):
    def parse(self):
        params = self.l.gaussian
        # mean, stdev, neuron_dims
        return {"op": "RandomNormal",
                "outputCount": 1,
                "attrs": {"dtype": lbann_onnx.ELEM_TYPE,
                          "mean": params.mean,
                          "scale": params.stdev,
                          "shape": params.neuron_dims if isinstance(params.neuron_dims, list) \
                          else list(map(int, params.neuron_dims.split(" ")))}}

class LbannLayerParser_reshape(LbannLayerParser):
    def parse(self):
        shape = list(map(int, self.l.reshape.dims.split(" ")))
        return {"op": "Reshape",
                "paramCount": 1,
                "inits": [{"shape": np.array(shape).shape,
                           "dataType": onnx.TensorProto.INT64,
                           "value": np.array(shape, dtype=np.int64).tobytes()}]}

class LbannLayerParser_reduction(LbannLayerParser):
    def parse(self):
        return {"op": {"sum": "ReduceSum",
                       "average": "ReduceMean"}[self.l.reduction.mode],
                "attrs": {"keepdims": 0}}

##
## Dummy parsers
##

class LbannLayerParser_evaluation(LbannLayerParser):
    def parse(self):
        return {"op": "LbannEvaluation"}

class LbannLayerParser_zero(LbannLayerParser):
    def parse(self):
        return {"op": "Identity"} # TODO: this is a dummy operation to perform correct infer_shape
