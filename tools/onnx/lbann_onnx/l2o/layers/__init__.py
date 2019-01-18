import re

class LbannLayerParser():
    def __init__(self, lp, inputShapes):
        self.lp = lp
        self.inputShapes = inputShapes

    def parse(self):
        raise NotImplementedError()

from lbann_onnx.l2o.layers.learnings    import *
from lbann_onnx.l2o.layers.math         import *
from lbann_onnx.l2o.layers.regularizers import *
from lbann_onnx.l2o.layers.transforms   import *
from lbann_onnx.l2o.layers.losses       import *
import lbann_onnx.l2o.layers as layers

# Parsers in a dict.
# PARSERS = {"abs": LbannLayerParser_abs, ...}
PARSERS = dict(map(lambda x: (x[0].group(1), getattr(layers, x[1])),
                  filter(lambda x: x[0] is not None,
                         map(lambda x: (re.compile("^LbannLayerParser_(.*)$").match(x), x),
                             dir()))))
