from lbann2onnx.util import parseSpatialAttributes, printError
from functools import reduce

import numpy as np

def parse_fully_connected(lp, inputShapes):
    outputSize = lp.num_neurons
    assert outputSize > 0

    wShape = [outputSize,
              int(np.prod(inputShapes[0][1:])) if len(inputShapes[0]) >= 2 else inputShapes[0][0]]
    bShape = [outputSize]
    return {"op": "Gemm",
            "paramCount": 2 if lp.has_bias else 1,
            "attrs": {"transB": 1},
            "params": [wShape, bShape] if lp.has_bias else [wShape]}

def parse_convolution(lp, inputShapes):
    attrs = parseSpatialAttributes(lp, "conv", True)
    attrs["group"] = lp.num_groups
    if attrs["group"] == 0:
        attrs["group"] = 1

    assert attrs["group"] == 1

    wShape = [lp.num_output_channels,
              inputShapes[0][1],
              *attrs["kernel_shape"]]
    bShape = [wShape[0]]

    return {"op": "Conv",
            "paramCount": 2 if lp.has_bias else 1,
            "attrs": attrs,
            "params": [wShape, bShape] if lp.has_bias else [wShape]}
