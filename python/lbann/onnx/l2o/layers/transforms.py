import lbann.onnx
from lbann.onnx.parserDescriptor import parserDescriptor
from lbann.onnx.util import getNodeAttributeByName
from lbann.onnx.l2o.util import parseSpatialAttributes
from lbann.onnx.l2o.layers import LbannLayerParser
import onnx
import numpy as np

@parserDescriptor(["MaxPool", "AveragePool"])
class LbannLayerParser_pooling(LbannLayerParser):
    def parse(self):
        params = self.l.pooling
        self.appendOperator({"max": "MaxPool",
                             "average": "AveragePool"}[params.pool_mode],
                            attrs=parseSpatialAttributes(params, "pool", False))

@parserDescriptor(["MaxUnpool"])
class LbannLayerParser_unpooling(LbannLayerParser):
    def parse(self):
        unpoolNode = list(filter(lambda x: x.name == self.l.unpooling.pooling_layer,
                                 self.knownNodes))
        assert len(unpoolNode) == 1
        self.appendOperator("MaxUnpool",
                            attrs=dict(map(lambda x: (x, getNodeAttributeByName(unpoolNode[0], x)),
                                           ["kernel_shape", "pads", "strides"])))

@parserDescriptor(["Split"])
class LbannLayerParser_slice(LbannLayerParser):
    def parse(self):
        params = self.l.slice
        offsets = list(map(int, params.slice_points.split(" ")))
        sizes = list(map(lambda x: offsets[x+1]-offsets[x], range(len(offsets)-1)))
        self.appendOperator("Split",
                            attrs={"axis": params.axis,
                                   "split": sizes})

@parserDescriptor(["Concat"])
class LbannLayerParser_concatenation(LbannLayerParser):
    def parse(self):
        self.appendOperator("Concat",
                            attrs={"axis": self.l.concatenation.axis})

@parserDescriptor(["RandomNormal"])
class LbannLayerParser_gaussian(LbannLayerParser):
    def parse(self):
        params = self.l.gaussian
        # mean, stdev, neuron_dims
        self.appendOperator("RandomNormal",
                            attrs={"dtype": lbann.onnx.ELEM_TYPE,
                                   "mean": params.mean,
                                   "scale": params.stdev,
                                   "shape": params.neuron_dims if isinstance(params.neuron_dims, list) \
                          else list(map(int, params.neuron_dims.split(" ")))})

@parserDescriptor(["Reshape"])
class LbannLayerParser_reshape(LbannLayerParser):
    def parse(self):
        shape = list(map(int, self.l.reshape.dims.split(" ")))
        h = self.createHiddenTensorName()
        self.appendOperator("Reshape", {}, 0, [self.getLbannInputNames()[0], h])
        self.appendParamWithInit(h, np.array(shape, dtype=np.int64))

@parserDescriptor(["ReduceSum", "ReduceMean"])
class LbannLayerParser_reduction(LbannLayerParser):
    def parse(self):
        self.appendOperator({"sum": "ReduceSum",
                             "average": "ReduceMean"}[self.l.reduction.mode],
                            attrs={"keepdims": 0})

##
## Dummy parsers
##

@parserDescriptor(stub=True)
class LbannLayerParser_evaluation(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannEvaluation")

@parserDescriptor(stub=True)
class LbannLayerParser_zero(LbannLayerParser):
    def parse(self):
        raise NotImplementedError()
