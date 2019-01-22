import sys
import numpy as np
import onnx
import google.protobuf.text_format as txtf

import lbann_pb2
import lbann_onnx.util
import lbann_onnx.o2l.layers as layers
from lbann_onnx.l2o import getTensorShapes

def onnxToLbannLayers(o, lbannInputNames, l2oInputMap, dataLayout="auto"):
    graph = o.graph
    tensorShapes = getTensorShapes(o, includeOutputShapes=True)
    opNames = list(map(getNodeName, enumerate(graph.node)))

    producers = {}
    for op, opName in zip(graph.node, opNames):
        for opt in op.output:
            assert not opt in producers.keys()
            producers[opt] = opName

    inputLayerName = "data"

    assert dataLayout == "auto"
    layers = []
    layers.append(lbann_pb2.Layer(name=inputLayerName,
                                  children=lbann_onnx.util.list2LbannList(lbannInputNames),
                                  data_layout="data_parallel",
                                  input=lbann_pb2.Input(io_buffer="partitioned")))
    for i in lbannInputNames:
        layers.append(lbann_pb2.Layer(name=i,
                                      parents=lbann_onnx.util.list2LbannList([inputLayerName]),
                                      data_layout="data_parallel",
                                      split=lbann_pb2.Split()))
        producers[i] = i
        if i in l2oInputMap.keys():
            producers[l2oInputMap[i]] = i

    for op, opName in zip(graph.node, opNames):
        inputShapes = list(map(lambda x: tensorShapes[x], op.input))
        outputShapes = list(map(lambda x: tensorShapes[x] if x in tensorShapes.keys() else None, op.output)) # Dropout's mask shape might be unknown
        inits = list(map(lambda x: getTensorInitial(x, graph), op.input))
        parents = list(map(lambda x: producers[x] if x in producers.keys() else None, op.input))
        layers.append(onnxNodeToLbannLayer(op, opName, inputShapes, outputShapes, inits, parents, dataLayout="auto"))

    return layers

def onnxNodeToLbannLayer(op, opName, inputShapes, outputShapes, inits, parents, dataLayout):
    opType = op.op_type
    parserName = "parse_{}".format(opType)
    if not hasattr(layers, parserName):
        print(lbann_onnx.util.printError("op_type \"{}\" is not supported.".format(opType)))
        assert False

    dic = getattr(layers, parserName)(op, inputShapes, outputShapes, inits).parse()

    validParents = list(filter(lambda x: x, parents)) # TODO: assert

    assert dataLayout == "auto"
    l = lbann_pb2.Layer(name=opName,
                        parents=lbann_onnx.util.list2LbannList(validParents),
                        data_layout=("model_parallel" if "fully_connected" in dic.keys() else "data_parallel"),
                        **dic)
    return l

def getNodeName(i_op):
    i, op = i_op
    return "{}_{}".format(op.op_type, i)

# TODO: move to util
def getTensorInitial(name, graph):
    for init in graph.initializer:
        if name == init.name:
            return np.frombuffer(init.raw_data, elemTypeToNumpy(init.data_type))

    return None

# TODO: move to util
def elemTypeToNumpy(t):
    if t == onnx.TensorProto.FLOAT:
        return np.float32
    elif t == onnx.TensorProto.UINT8:
        return np.uint8
    elif t == onnx.TensorProto.INT8:
        return np.int8
    elif t == onnx.TensorProto.UINT16:
        return np.uint16
    elif t == onnx.TensorProto.INT16:
        return np.int16
    elif t == onnx.TensorProto.INT32:
        return np.int32
    elif t == onnx.TensorProto.INT64:
        return np.int64
    elif t == onnx.TensorProto.STRING:
        return np.str
    elif t == onnx.TensorProto.BOOL:
        return np.bool
    assert False
