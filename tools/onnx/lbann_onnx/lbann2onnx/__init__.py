#!/usr/bin/env python3

from lbann_pb2 import LbannPB, Model

import google.protobuf.text_format as txtf
import onnx
import onnx.shape_inference
import numpy as np
from functools import reduce
import sys

import lbann_onnx
from lbann_onnx.lbann2onnx.functions import parseLbannLayer

def getTensorShapes(o):
    o = onnx.shape_inference.infer_shapes(o)

    # infer Split-ted shapes manually (https://github.com/onnx/onnx/issues/1735)
    while False:
        hasSplit = False
        for n in o.graph.node:
            if n.op_type == "Split":
                splits = list(filter(lambda x: x.name == "split", n.attribute))
                if len(splits) == 0:
                    continue

                assert len(splits) == 1
                split = splits[0].ints
                axes = list(filter(lambda x: x.name == "axis", n.attribute))
                axis = axes[0].i if len(axes) == 1 else 0

                inputVI = list(filter(lambda x: x.name == n.input[0], list(o.graph.value_info) + list(o.graph.input)))[0]
                inputShape = getDimFromValueInfo(inputVI)
                outputShapes = list(map(lambda x: x.shape, np.split(np.zeros(inputShape), np.cumsum(split)[:-1], axis)))
                vis = list(o.graph.value_info)
                for outputName, outputShape in zip(n.output, outputShapes):
                    vis.append(onnx.helper.make_tensor_value_info(name=outputName,
                                                                  elem_type=inputVI.type.tensor_type.elem_type,
                                                                  shape=outputShape))

                g = onnx.helper.make_graph(o.graph.node,
                                           o.graph.input,
                                           o.graph.output,
                                           o.graph.initializer,
                                           vis)
                o = onnx.helper.make_model(g)

                hasSplit = True

        if hasSplit:
            o = onnx.shape_inference.infer_shapes(o)

        else:
            break

    vis = o.graph.value_info
    vis.extend(o.graph.input)
    return dict(map(lambda x: (x.name,
                               list(map(lambda y: y.dim_value, x.type.tensor_type.shape.dim))),
                    vis))

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
        inputShapes = getTensorShapes(onnx.helper.make_model(onnx.helper.make_graph(nodes, "graph",
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
                l.gaussian.neuron_dims = " ".join(map(str, dims))

            else:
                lbann_onnx.util.printError("\"hint_layer\" is supported only for fully_connected or gaussian.")
                exit()

        if l.num_neurons_from_data_reader:
            if not l.HasField("fully_connected"):
                lbann_onnx.util.printError("\"num_neurons_from_data_reader\" in non-fully-connected layers are not supported.")
                exit()

            assert len(inputs) > 0
            dims = lbann_onnx.util.getDimFromValueInfo(inputs[0])
            assert len(dims) > 1 and dims[0] == miniBatchSize
            l.fully_connected.num_neurons = int(np.prod(dims[1:]))

        ret = parseLbannLayer(l, inputShapes, nodes)
        if "inputs" in ret.keys():
            inputs.extend(ret["inputs"])

        if "inits" in ret.keys():
            inits.extend(ret["inits"])

        if "node" in ret.keys():
            if ret["node"].op_type in ["Gemm"]:
                gemm = ret["node"]

                for i_input in range(len(gemm.input)):
                    if gemm.op_type == "Gemm" and i_input != 0:
                        continue

                    nameBeforeReshape = gemm.input[i_input]
                    shapeBeforeReshape = inputShapes[nameBeforeReshape]

                    if len(shapeBeforeReshape) != 2:
                        nameAfterReshape = "{}_reshaped_{}".format(nameBeforeReshape, len(nodes))
                        if len(shapeBeforeReshape) > 2:
                            shapeAfterReshape = [shapeBeforeReshape[0], int(np.prod(shapeBeforeReshape[1:]))]
                        else:
                            shapeAfterReshape = [1, shapeBeforeReshape[0]]

                        shapeName = "{}_shape_{}".format(nameBeforeReshape, len(nodes))

                        reshape = onnx.helper.make_node("Reshape",
                                                        inputs=[nameBeforeReshape, shapeName],
                                                        outputs=[nameAfterReshape])
                        shapeInit = onnx.helper.make_tensor(name=shapeName,
                                                            data_type=onnx.TensorProto.INT64,
                                                            dims=[2],
                                                            vals=np.array(shapeAfterReshape, dtype=np.int64).tobytes(),
                                                            raw=True)

                        nodes.append(reshape)
                        inits.append(shapeInit)

                        gemm.input[i_input] = nameAfterReshape

            nodes.append(ret["node"])

    for l in params.keys():
        for i,p in enumerate(params[l]):
            name = "{}_p{}".format(l, i)
            inits.append(onnx.helper.make_tensor(name=name,
                                                 data_type=lbann_onnx.ELEM_TYPE,
                                                 dims=p.shape,
                                                 vals=p.tobytes(),
                                                 raw=True))

    for metric in pb.model.metric:
        assert metric.HasField("layer_metric")
        outputs.append(onnx.helper.make_tensor_value_info(name="{}_0".format(metric.layer_metric.layer),
                                                          elem_type=lbann_onnx.ELEM_TYPE,
                                                          shape=[]))

    for term in pb.model.objective_function.layer_term:
        outputs.append(onnx.helper.make_tensor_value_info(name="{}_0".format(term.layer),
                                                          elem_type=lbann_onnx.ELEM_TYPE,
                                                          shape=[]))

    g = onnx.helper.make_graph(nodes, "graph", inputs, outputs, inits)
    o = onnx.helper.make_model(g)
    if addValueInfo:
        o = onnx.shape_inference.infer_shapes(o)

    return o, miniBatchSize
