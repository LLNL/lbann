import numpy as np

import lbann_onnx.util
from lbann_onnx.util import getNodeAttributeByName, list2LbannList
import lbann_pb2

class OnnxLayerParser():
    def __init__(self, op, inputShapes, outputShapes, inits):
        self.op = op
        self.inputShapes = inputShapes
        self.outputShapes = outputShapes
        self.inits = inits

    def parse(self):
        raise NotImplementedError()

    def getNodeAttribute(self, attr, defVal=None):
        return getNodeAttributeByName(self.op, attr, defVal)

    def parseAttrList(self, attr, defVal=None):
        l = self.getNodeAttribute(attr, defVal)
        return list2LbannList(l)

class OnnxSpatialLayerParser(OnnxLayerParser):
    def parse_Spatial(self, num_dims, prefix, hasDilations):
        dic = {"dims": self.parseAttrList("kernel_shape"),
               "strides": self.parseAttrList("strides"),
               "pads": list2LbannList(lbann_onnx.util.getOneSidePads(self.getNodeAttribute("pads", [0]*(num_dims*2))))}
        if hasDilations:
            dic["dilations"] = self.parseAttrList("dilations", [1]*num_dims)

        dic = dict(map(lambda x: ("{}_{}".format(prefix, x), dic[x]), dic.keys()))
        dic["num_dims"] = num_dims
        return dic

from lbann_onnx.o2l.layers.learnings import *
from lbann_onnx.o2l.layers.math import *
from lbann_onnx.o2l.layers.regularizers import *
from lbann_onnx.o2l.layers.transforms import *
