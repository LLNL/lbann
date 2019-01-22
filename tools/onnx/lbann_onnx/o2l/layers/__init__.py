import numpy as np

import lbann_onnx.util
from lbann_onnx.util import getNodeAttributeByName, list2LbannList
import lbann_pb2

# TODO: split this file like l2o

class OnnxLayerParser():
    def __init__(self, op, inputShapes, outputShapes, inits):
        self.op = op
        self.inputShapes = inputShapes
        self.outputShapes = outputShapes
        self.inits = inits

    def parse(self):
        raise NotImplementedError()

    def getNodeAttribute(self, attr, defVal=None):
        return getNodeAttributeByName(self.op, attr, defVal, typeConversion=True)

    def parseAttrList(self, attr, defVal=None):
        l = self.getNodeAttribute(attr, defVal)
        return list2LbannList(l)

class parse_Conv(OnnxLayerParser):
    def parse(self):
        num_dims = len(self.inputShapes[0])-2
        group = self.getNodeAttribute("group", 1)
        convolution = lbann_pb2.Convolution(
            num_dims = num_dims,
            num_output_channels = self.outputShapes[0][1],
            num_groups = group,
            conv_dims = self.parseAttrList("kernel_shape"),
            conv_strides = self.parseAttrList("strides"),
            conv_dilations = self.parseAttrList("dilations", [1]*num_dims),
            conv_pads = list2LbannList(getOneSidePads(self.getNodeAttribute("pads", [0]*(num_dims*2)))),
            has_bias = (len(self.inputShapes) == 3),
        )
        return {"convolution": convolution}

class parse_Relu(OnnxLayerParser):
    def parse(self):
        return {"relu": lbann_pb2.Relu()}

class _parse_MaxAveragePool(OnnxLayerParser):
    def parse_MaxAveragePool(self, average):
        num_dims = len(self.inputShapes[0])-2
        pooling = lbann_pb2.Pooling(
            num_dims = num_dims,
            pool_dims = self.parseAttrList("kernel_shape"), # TODO: merge to parse_Conv
            pool_strides = self.parseAttrList("strides"),
            pool_pads = list2LbannList(getOneSidePads(self.getNodeAttribute("pads", [0]*(num_dims*2)))),
            pool_mode = "average" if average else "max",
        )
        return {"pooling": pooling}

class parse_MaxPool(_parse_MaxAveragePool):
    def parse(self):
        return self.parse_MaxAveragePool(average=False)

class parse_AveragePool(_parse_MaxAveragePool):
    def parse(self):
        return self.parse_MaxAveragePool(average=True)

# TODO: return identity if not necessary
class parse_Reshape(OnnxLayerParser):
    def parse(self):
        reshape = lbann_pb2.Reshape(
            num_dims = len(self.inputShapes[1]),
            dims = list2LbannList(self.inits[1])
        )
        return {"reshape": reshape}

class parse_Gemm(OnnxLayerParser):
    def parse(self):
        assert self.getNodeAttribute("transA",0) == 0 and self.getNodeAttribute("transB",0) == 1
        assert self.getNodeAttribute("alpha",1.0) == 1.0 and (self.getNodeAttribute("beta",1.0) == 1.0 or len(self.inputShapes) < 3)
        # TODO: transform
        fully_connected = lbann_pb2.FullyConnected(
            num_neurons = self.outputShapes[0][1],
            has_bias = (len(self.inputShapes) == 3)
        )
        return {"fully_connected": fully_connected}

class parse_MatMul(OnnxLayerParser):
    def parse(self):
        fully_connected = lbann_pb2.FullyConnected(
            num_neurons = self.outputShapes[0][1],
            has_bias = False
        )
        return {"fully_connected": fully_connected}

class parse_LRN(OnnxLayerParser):
    def parse(self):
        local_response_normalization = lbann_pb2.LocalResponseNormalization(
            lrn_alpha = self.getNodeAttribute("alpha"),
            lrn_beta = self.getNodeAttribute("beta"),
            lrn_k = self.getNodeAttribute("bias"),
            window_width = self.getNodeAttribute("size"),
        )
        return {"local_response_normalization": local_response_normalization}

class parse_BatchNormalization(OnnxLayerParser):
    def parse(self):
        batch_normalization = lbann_pb2.BatchNormalization(
            epsilon = self.getNodeAttribute("epsilon", 1e-5),
            decay = self.getNodeAttribute("momentum", 0.9),
        )
        return {"batch_normalization": batch_normalization}

class parse_Dropout(OnnxLayerParser):
    def parse(self):
        return {"dropout": lbann_pb2.Dropout(keep_prob = 1.0-self.getNodeAttribute("ratio"))}

class parse_Softmax(OnnxLayerParser):
    def parse(self):
        return {"softmax": lbann_pb2.Softmax()}

class parse_Concat(OnnxLayerParser):
    def parse(self):
        return {"concatenation": lbann_pb2.Concatenation(concatenation_axis = self.getNodeAttribute("axis"))}

class parse_Sum(OnnxLayerParser):
    def parse(self):
        return {"sum": lbann_pb2.Sum()}

class parse_Add(OnnxLayerParser):
    def parse(self):
        return {"add": lbann_pb2.Add()}

# TODO: move to util
def getOneSidePads(pads, assertEvens=False):
    # [s1, s2, ..., e1, e2, ...] -> [s1, s2, ...]
    assert len(pads)%2 == 0
    count = int(len(pads)/2)

    begins = pads[:count]
    ends   = pads[count:]
    if not begins == ends:
        assert not assertEvens
        d = set(np.array(ends)-np.array(ends))
        assert d == set([0]) or d == set([0, 1]) # accept |p_end - p_begin| = 0 or 1
        lbann_onnx.util.printError("Padding widths of at least one dimension is not the same: {}".format(pads))

    return begins
