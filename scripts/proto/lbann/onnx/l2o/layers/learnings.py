from lbann.onnx.parserDescriptor import parserDescriptor
from lbann.onnx.l2o.util import parseSpatialAttributes
from lbann.onnx.l2o.layers import LbannLayerParser
from functools import reduce

import numpy as np
import onnx

@parserDescriptor(["Reshape", "Gemm"])
class LbannLayerParser_fully_connected(LbannLayerParser):
    def parse(self):
        gemmInput, = self.getLbannInputNames()
        shape = self.inputShapes[0]
        if len(shape) != 2:
            if len(shape) > 2:
                shape = [shape[0], int(np.prod(shape[1]))]
            else:
                shape = [1, shape[0]]

            h1 = self.createHiddenTensorName()
            h2,_ = self.appendOperator("Reshape", {}, 0, [self.getLbannInputNames()[0], h1], 1)
            self.appendParamWithInit(h1, data=np.array(shape, dtype=np.int64))
            gemmInput = h2

        params = self.l.fully_connected
        outputSize = params.num_neurons
        assert outputSize > 0

        wShape = [outputSize, shape[1]]
        bShape = [outputSize]

        outputs, paramNames = self.appendOperator("Gemm",
                                                  {"transB": 1},
                                                  2 if params.has_bias else 1,
                                                  [gemmInput, self.getParamName(0), self.getParamName(1)] \
                                                  if params.has_bias else [gemmInput, self.getParamName(0)])
        self.appendParam(paramNames[0], wShape)
        if params.has_bias:
            self.appendParam(paramNames[1], bShape)

@parserDescriptor(["Conv"])
class LbannLayerParser_convolution(LbannLayerParser):
    def parse(self):
        params = self.l.convolution
        attrs = parseSpatialAttributes(params, "conv", True)
        attrs["group"] = params.num_groups
        if attrs["group"] == 0:
            attrs["group"] = 1

        assert attrs["group"] == 1

        wShape = [params.num_output_channels,
                  self.inputShapes[0][1],
                  *attrs["kernel_shape"]]
        bShape = [wShape[0]]

        outputs, paramNames = self.appendOperator("Conv",
                                                  attrs,
                                                  2 if params.has_bias else 1)
        self.appendParam(paramNames[0], wShape)
        if params.has_bias:
            self.appendParam(paramNames[1], bShape)
