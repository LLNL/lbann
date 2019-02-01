import re
import numpy as np

import lbann.onnx.util
from lbann.onnx.util import getNodeAttributeByName, list2LbannList

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
               "pads": list2LbannList(lbann.onnx.util.getOneSidePads(self.getNodeAttribute("pads", [0]*(num_dims*2))))}
        if hasDilations:
            dic["dilations"] = self.parseAttrList("dilations", [1]*num_dims)

        dic = dict(map(lambda x: ("{}_{}".format(prefix, x), dic[x]), dic.keys()))
        dic["has_vectors"] = True
        dic["num_dims"] = num_dims
        return dic

from lbann.onnx.o2l.layers.learnings import *
from lbann.onnx.o2l.layers.math import *
from lbann.onnx.o2l.layers.regularizers import *
from lbann.onnx.o2l.layers.transforms import *
import lbann.onnx.o2l.layers as layers

PARSERS = dict(map(lambda x: (x, getattr(layers, "parse_{}".format(x))),
                   map(lambda x: x.group(1),
                       filter(lambda x: x is not None,
                              map(lambda x: re.compile("parse_(.+)$").match(x),
                                  dir(layers))))))
