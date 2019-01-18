from lbann_onnx.util import printError
from lbann_onnx.l2o.util import parseSpatialAttributes
from lbann_onnx.l2o.layers import LbannLayerParser
from functools import reduce

import numpy as np

class LbannLayerParser_fully_connected(LbannLayerParser):
    def parse(self):
        params = self.l.fully_connected
        outputSize = params.num_neurons
        assert outputSize > 0

        wShape = [outputSize,
                  int(np.prod(self.inputShapes[0][1:])) if len(self.inputShapes[0]) >= 2 else self.inputShapes[0][0]]
        bShape = [outputSize]
        return {"op": "Gemm",
                "paramCount": 2 if params.has_bias else 1,
                "attrs": {"transB": 1},
                "params": [wShape, bShape] if params.has_bias else [wShape]}

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

        return {"op": "Conv",
                "paramCount": 2 if params.has_bias else 1,
                "attrs": attrs,
                "params": [wShape, bShape] if params.has_bias else [wShape]}
