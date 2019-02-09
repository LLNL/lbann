import os
import sys
import onnx
import subprocess
import numpy as np

def getLbannRoot():
    env = os.getenv("LBANN_ROOT")
    if env is not None:
        return env

    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")

def printWarning(s):
    if parseBoolEnvVar("LBANN_ONNX_VERBOSE", False):
        sys.stderr.write("lbann-onnx warning: {}\n".format(s))

def printParsingState(node, knownShapes):
    printWarning("Operation: \n\n{}".format(node))
    printWarning("The list of known shapes:")
    for n, s in knownShapes.items():
        printWarning("   {:10} {}".format(n, tuple(s)))

def getDimFromValueInfo(vi):
    return list(map(lambda x: x.dim_value, vi.type.tensor_type.shape.dim))

def getNodeAttributeByName(node, attr, defVal=None):
    ret = list(filter(lambda x: x.name == attr, node.attribute))
    if len(ret) != 1:
        if defVal is not None:
            return defVal

        assert False

    v = ret[0]
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

def list2LbannList(l):
    return " ".join(map(str, l))

def lbannList2List(l):
    return list(map(int, l.split(" ")))

def getOneSidePads(pads, assertEvens=False):
    # [s1, s2, ..., e1, e2, ...] -> [s1, s2, ...]
    assert len(pads)%2 == 0
    count = int(len(pads)/2)

    begins = pads[:count]
    ends   = pads[count:]
    if not begins == ends:
        assert not assertEvens
        d = set(np.array(ends)-np.array(ends))
        assert d == set([0]) or d == set([0, 1]) # accept |p_end - p_begin| = 0 or 1
        printWarning("Padding widths of at least one dimension is not the same: {}".format(pads))

    return begins

def getStaticTensorShapes(o):
    o = onnx.shape_inference.infer_shapes(o)
    vis = o.graph.value_info
    vis.extend(o.graph.input)
    vis.extend(o.graph.output)
    return dict(map(lambda x: (x.name,
                               list(map(lambda y: y.dim_value, x.type.tensor_type.shape.dim))),
                    vis))

def parseBoolEnvVar(name, defVal):
    if not name in os.environ.keys():
        return defVal

    v = os.environ[name]
    return v == "1"
