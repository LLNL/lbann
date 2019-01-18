import re
import onnx.helper

class LbannLayerParser():
    def __init__(self, l, layerType, inputShapes):
        self.l = l
        self.layerType = layerType
        self.inputShapes = inputShapes

        self.nodes = []

        # TODO: rename
        self.inputs = []
        self.inits = []

    def parse(self):
        raise NotImplementedError()

    def appendOperator(self, op, attrs={}, paramShapes=[], outputCount=1):
        lbannInputs = list(map(lambda x: "{}_0".format(x),
                               self.l.parents.split(" ") if self.l.parents != "" else []))
        lbannOutputs = self.l.children.split(" ") if len(self.l.children) > 0 else []
        paramNames = list(map(self.getParamName, range(len(self.inputs)+len(paramShapes))))
        inputNames  = lbannInputs + paramNames
        outputNames = list(map(lambda x: "{}_{}".format(self.l.name, x), range(outputCount))) \
            if len(lbannOutputs) == 0 else list(map(lambda x: "{}_0".format(x), lbannOutputs))

        node = onnx.helper.make_node(op,
                                     inputs=inputNames,
                                     outputs=outputNames,
                                     name=self.l.name,
                                     lbannOp=self.layerType,
                                     lbannDataLayout=self.l.data_layout,
                                     **attrs)
        self.nodes.append(node)

        for n, s in zip(paramNames, paramShapes):
            i = onnx.helper.make_tensor_value_info(name=n,
                                                   elem_type=lbann_onnx.ELEM_TYPE,
                                                   shape=s)
            self.inputs.append(i)

        return paramNames

    def appendInit(self, name, shape, dataType, data):
        init = onnx.helper.make_tensor(name=name,
                                       dims=shape,
                                       data_type=dataType,
                                       vals=data,
                                       raw=True) # OPTIMIZE
        self.inits.append(init)

    def getParamName(self, i):
        return "{}_p{}".format(self.l.name, i)

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
