import re

from lbann_onnx.l2o.layers.learnings    import *
from lbann_onnx.l2o.layers.math         import *
from lbann_onnx.l2o.layers.regularizers import *
from lbann_onnx.l2o.layers.transforms   import *
from lbann_onnx.l2o.layers.losses       import *
import lbann_onnx.l2o.layers as layers

# Parser layers in a dict.
# LAYERS = {"abs": parse_abs, ...}
LAYERS = dict(map(lambda x: (x[0].group(1), getattr(layers, x[1])),
                  filter(lambda x: x[0] is not None,
                         map(lambda x: (re.compile("^parse_(.*)$").match(x), x),
                             dir()))))
