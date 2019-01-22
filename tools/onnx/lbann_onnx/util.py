import sys
import onnx

def printError(s):
    sys.stderr.write("lbann-onnx error: {}\n".format(s))

def printParsingState(node, knownShapes):
    printError("Operation: \n\n{}".format(node))
    printError("The list of known shapes:")
    for n, s in knownShapes.items():
        printError("   {:10} {}".format(n, tuple(s)))

def getDimFromValueInfo(vi):
    return list(map(lambda x: x.dim_value, vi.type.tensor_type.shape.dim))

# TODO: type check
def getNodeAttributeByName(node, attr, defVal=None, typeConversion=False):
    ret = list(filter(lambda x: x.name == attr, node.attribute))
    if len(ret) != 1:
        if defVal is not None:
            return defVal

        assert False

    v = ret[0]
    if not typeConversion:
        return v

    t = v.type
    if t == onnx.AttributeProto.INTS:
        return v.ints
    elif t == onnx.AttributeProto.FLOAT:
        return v.f
    elif t == onnx.AttributeProto.INT:
        return v.i
    elif t == onnx.AttributeProto.STRING:
        return v.s.decode("utf-8")

    assert False

# TODO: replace old list2LbannList expressions
def list2LbannList(l):
    return " ".join(map(str, l))
