#!/usr/bin/env python3

from lbann_pb2 import LbannPB, Model

import google.protobuf.text_format as txtf
import onnx
import onnx.shape_inference
import numpy as np
from functools import reduce
import sys

import lbann.onnx
from lbann.onnx.l2o.layers import PARSERS
from lbann.onnx.util import list2LbannList, getStaticTensorShapes

def parseLbannModelPB(path, modelInputShapes, params={}, addValueInfo=True):
    with open(path, "r") as f:
        s = f.read().strip()

    pb = LbannPB()
    txtf.Merge(s, pb)

    miniBatchSize = pb.model.mini_batch_size
    assert isinstance(miniBatchSize, int) and miniBatchSize > 0
    for k in modelInputShapes.keys():
        modelInputShapes[k] = tuple([miniBatchSize] + list(modelInputShapes[k]))

    nodes = []
    inputs = []
    outputs = []
    inits = []

    for l in pb.model.layer:
        # OPTIMIZE: avoid performing infer_shapes in every iteration. value_info can be passed via make_graph
        inputShapes = getStaticTensorShapes(onnx.helper.make_model(onnx.helper.make_graph(nodes, "graph",
                                                                                          inputs, outputs, inits)))
        inputShapes.update(modelInputShapes)
        inputShapes.update(dict(map(lambda x: (x.name, x.dims), inits)))

        tensorNames = set(reduce(lambda a,b: a+b, list(map(lambda x: list(x.input)+list(x.output), nodes)), []))
        unknownTensors = list(filter(lambda x: x not in inputShapes.keys(), tensorNames))
        # if len(unknownTensors) > 0:
        #     print(unknownTensors)
        #     assert False

        # iterative export for debugging
        # onnx.save(onnx.shape_inference.infer_shapes(onnx.helper.make_model(onnx.helper.make_graph(nodes,
        #                                                                                           "graph",
        #                                                                                           inputs,
        #                                                                                           outputs,
        #                                                                                           inits))),
        #           "out_{}.onnx".format(len(nodes)))

        if l.hint_layer:
            dims = None

            hintLayer = list(filter(lambda x: x.name == l.hint_layer, nodes))
            if len(hintLayer) > 0:
                assert len(hintLayer) == 1
                dims = inputShapes[hintLayer[0].output[0]]

            elif l.hint_layer in inputShapes.keys():
                dims = inputShapes[l.hint_layer]

            assert dims is not None

            if l.HasField("fully_connected"):
                assert len(dims) > 1 and dims[0] == miniBatchSize
                l.fully_connected.num_neurons = int(np.prod(dims[1:]))

            elif l.HasField("gaussian"):
                l.gaussian.neuron_dims = list2LbannList(dims)

            else:
                lbann.onnx.util.printWarning("\"hint_layer\" is supported only for fully_connected or gaussian.")
                exit()

        if l.num_neurons_from_data_reader:
            if not l.HasField("fully_connected"):
                lbann.onnx.util.printWarning("\"num_neurons_from_data_reader\" in non-fully-connected layers are not supported.")
                exit()

            assert len(inputs) > 0
            dims = lbann.onnx.util.getDimFromValueInfo(inputs[0])
            assert len(dims) > 1 and dims[0] == miniBatchSize
            l.fully_connected.num_neurons = int(np.prod(dims[1:]))

        ret = parseLbannLayer(l, inputShapes, nodes)
        if "inputs" in ret.keys():
            inputs.extend(ret["inputs"])

        if "inits" in ret.keys():
            inits.extend(ret["inits"])

        if "nodes" in ret.keys():
            nodes.extend(ret["nodes"])

    for l in params.keys():
        for i,p in enumerate(params[l]):
            name = "{}_p{}".format(l, i)
            inits.append(onnx.math_helper.to_array(p, name=name))

    for metric in pb.model.metric:
        assert metric.HasField("layer_metric")
        outputs.append(onnx.helper.make_tensor_value_info(name="{}_0".format(metric.layer_metric.layer),
                                                          elem_type=lbann.onnx.ELEM_TYPE,
                                                          shape=[]))

    for term in pb.model.objective_function.layer_term:
        outputs.append(onnx.helper.make_tensor_value_info(name="{}_0".format(term.layer),
                                                          elem_type=lbann.onnx.ELEM_TYPE,
                                                          shape=[]))

    g = onnx.helper.make_graph(nodes, "graph", inputs, outputs, inits)
    o = onnx.helper.make_model(g)
    if addValueInfo:
        o = onnx.shape_inference.infer_shapes(o)

    return o, miniBatchSize

def parseLbannLayer(l, tensorShapes, knownNodes):
    if any(map(lambda x: l.HasField(x), ["input",
                                         "identity", # LBANN's "identity" does not have outputs
                                         "dummy"])):
        return {}

    if l.HasField("split"):
        if l.name not in tensorShapes.keys():
            lbann.onnx.util.printWarning("The shape of \"{}\" cannot be inferred.".format(l.name) \
                                       + " This error may happen when you set incorret an input tensor name.")
            lbann.onnx.util.printParsingState(l, tensorShapes)
            exit()

        ipt = onnx.helper.make_tensor_value_info(name="{}_0".format(l.name),
                                                 elem_type=lbann.onnx.ELEM_TYPE,
                                                 shape=tensorShapes[l.name])

        return {"inputs": [ipt]}

    lbannInputs = list(map(lambda x: "{}_0".format(x),
                           l.parents.split(" ") if l.parents != "" else []))
    lbannOutputs = l.children.split(" ") if len(l.children) > 0 else []

    for f in PARSERS.keys():
        if l.HasField(f):
            for i in lbannInputs:
                if not i in tensorShapes.keys():
                    lbann.onnx.util.printWarning("The shape of \"{}\" cannot be inferred.".format(i))
                    lbann.onnx.util.printParsingState(l, tensorShapes)
                    exit()

            p = PARSERS[f](l,
                           f,
                           list(map(lambda x: tensorShapes[x], lbannInputs)),
                           knownNodes)
            p.parse()
            return {"nodes": p.nodes, "inputs": p.inputs, "inits": p.inits}

    NotImplementedError("Unimplemented LBANN operator: {}".format(l))
