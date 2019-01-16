import sys

##
## Debugging functions
##

# TODO: move to lbann-onnx but not lbann2onnx
def printError(s):
    sys.stderr.write("lbann-onnx error: {}\n".format(s))

def printParsingState(node, knownShapes):
    printError("Operation: \n\n{}".format(node))
    printError("The list of known shapes:")
    for n, s in knownShapes.items():
        printError("   {:10} {}".format(n, tuple(s)))

def getDimFromValueInfo(vi):
    return list(map(lambda x: x.dim_value, vi.type.tensor_type.shape.dim))

def getNodeAttributeByName(node, attr):
    ret = list(filter(lambda x: x.name == attr, node.attribute))
    assert len(ret) == 1
    return ret[0]

##
## Parsing helper functions
##

# OPTIMIZE: use has_vectors
def parseSpatialAttribute(params, attr, dims):
    ary = getattr(params, attr)
    if ary != "":
        return list(map(int, ary.split(" ")))
    else:
        return [getattr(params, "{}_i".format(attr))]*dims

def parseSpatialAttributes(lp, name, hasDilations):
    spatialParamNameMap = {"kernel_shape": "dims",
                           "pads":         "pads",
                           "strides":      "strides"}
    if hasDilations:
        spatialParamNameMap["dilations"] = "dilations"

    attrs = dict(map(lambda x: (x,
                                parseSpatialAttribute(lp,
                                                      "{}_{}".format(name, spatialParamNameMap[x]),
                                                      lp.num_dims)),
                     spatialParamNameMap.keys()))

    assert len(attrs["pads"]) == lp.num_dims

    attrs["pads"] *= 2
    if hasDilations:
        attrs["dilations"] = list(map(lambda x: 1 if x == 0 else x, attrs["dilations"]))

    return attrs
