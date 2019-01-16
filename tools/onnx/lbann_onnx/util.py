import sys

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
