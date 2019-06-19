import re
import onnx.helper
from onnx import numpy_helper

class LbannLayerParser():
    def __init__(self, l, layerType, inputShapes, knownNodes):
        self.l = l
        self.layerType = layerType
        self.inputShapes = inputShapes
        self.knownNodes = knownNodes # necessary to get existing nodes' information for unpooling

        self.nodes = []
        self.paramValueInfos = []
        self.paramInits = []

        self.hiddenTensorCount = 0

    def parse(self):
        raise NotImplementedError()

    def getLbannInputNames(self):
        return list(map(lambda x: "{}_0".format(x),
                        self.l.parents.split(" ") if self.l.parents != "" else []))

    def getLbannOutputNames(self):
        return self.l.children.split(" ") if len(self.l.children) > 0 else []

    def appendOperator(self, op, attrs={}, paramCount=0, inputNames=None, hiddenOutputCount=None):
        paramNames = list(map(self.getParamName, range(len(self.paramValueInfos)+paramCount)))

        if inputNames is None:
            inputNames  = self.getLbannInputNames() + paramNames

        if hiddenOutputCount is None:
            lbannOutputs = self.getLbannOutputNames()
            outputNames = list(map(lambda x: "{}_{}".format(self.l.name, x), range(1))) \
                if len(lbannOutputs) == 0 else list(map(lambda x: "{}_0".format(x), lbannOutputs))

        else:
            outputNames = list(map(lambda x: self.createHiddenTensorName(), range(hiddenOutputCount)))

        node = onnx.helper.make_node(op,
                                     inputs=inputNames,
                                     outputs=outputNames,
                                     name=self.l.name,
                                     lbannOp=self.layerType,
                                     lbannDataLayout=self.l.data_layout,
                                     **attrs)
        self.nodes.append(node)
        return outputNames[0], paramNames

    def appendParam(self, name, shape):
        i = onnx.helper.make_tensor_value_info(name=name,
                                               elem_type=lbann.onnx.ELEM_TYPE,
                                               shape=shape)
        self.paramValueInfos.append(i)

    def appendParamWithInit(self, name, data):
        self.appendParam(name, data.shape)
        init = onnx.numpy_helper.from_array(data, name=name)
        self.paramInits.append(init)

    def getParamName(self, i):
        return "{}_p{}".format(self.l.name, i)

    def createHiddenTensorName(self):
        n = "{}_h{}".format(self.l.name, self.hiddenTensorCount)
        self.hiddenTensorCount += 1
        return n


from lbann.onnx.l2o.layers.learnings    import *
from lbann.onnx.l2o.layers.math         import *
from lbann.onnx.l2o.layers.regularizers import *
from lbann.onnx.l2o.layers.transforms   import *
from lbann.onnx.l2o.layers.losses       import *
import lbann.onnx.l2o.layers as layers

# Parsers in a dict.
# PARSERS = {"abs": LbannLayerParser_abs, ...}
PARSERS = dict(map(lambda x: (x[0].group(1), getattr(layers, x[1])),
                   filter(lambda x: x[0] is not None,
                          map(lambda x: (re.compile("^LbannLayerParser_(.*)$").match(x), x),
                              dir()))))
