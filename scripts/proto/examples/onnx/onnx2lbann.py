#!/usr/bin/env python3

"""
Convert an ONNX model to a LBANN model
Run "./onnx2lbann.py --help" for more details.
"""

import argparse
import onnx
import google.protobuf.text_format as txtf

import lbann.onnx.o2l
from lbann.lbann_proto import lbann_pb2

dataLayerName = "image"
labelLayerName = "label"
probLayerName = "prob"
crossEntropyLayerName = "cross_entropy"
top1AccuracyLayerName = "top1_accuracy"
top5AccuracyLayerName = "top5_accuracy"

def convertAndSave(path, outputPath):
    o = onnx.load(path)
    onnxInputLayer, = set(map(lambda x: x.name, o.graph.input)) - set(map(lambda x: x.name, o.graph.initializer))

    # Parse layers
    layers = lbann.onnx.o2l.onnxToLbannLayers(
        o,
        [dataLayerName, labelLayerName],
        {dataLayerName: onnxInputLayer},
    )

    # Add a softmax layer
    outputLayerName = layers[-1].name
    probLayerName = outputLayerName
    # layers.append(lbann_pb2.Layer(name=probLayerName,
    #                               parents=lbann.onnx.util.list2LbannList([outputLayerName]),
    #                               data_layout="data_parallel",
    #                               softmax=lbann_pb2.Softmax()))

    # Add metric layers
    for name, dic in [(crossEntropyLayerName, {"cross_entropy": lbann_pb2.CrossEntropy()}),
                      (top1AccuracyLayerName, {"categorical_accuracy": lbann_pb2.CategoricalAccuracy()}),
                      (top5AccuracyLayerName, {"top_k_categorical_accuracy": lbann_pb2.TopKCategoricalAccuracy(k=5)})]:
        layers.append(lbann_pb2.Layer(name=name,
                                      parents=lbann.onnx.util.list2LbannList([probLayerName, labelLayerName]),
                                      data_layout="data_parallel",
                                      **dic))

    # Define an objective function
    objective = lbann_pb2.ObjectiveFunction(
        layer_term = [lbann_pb2.LayerTerm(layer=crossEntropyLayerName)],
        l2_weight_regularization = [lbann_pb2.L2WeightRegularization(scale_factor=1e-4)]
    )

    # Add metrics
    metrics = []
    for name, layer, unit in [("categorical accuracy", top1AccuracyLayerName, "%"),
                              ("top-5 categorical accuracy", top5AccuracyLayerName, "%")]:
        metrics.append(lbann_pb2.Metric(layer_metric=lbann_pb2.LayerMetric(name=name,
                                                                           layer=layer,
                                                                           unit=unit)))

    # Add callbacks
    callbacks = []
    for dic in [{"print": lbann_pb2.CallbackPrint()},
                {"timer": lbann_pb2.CallbackTimer()},
                {"imcomm": lbann_pb2.CallbackImComm(intermodel_comm_method="normal", all_optimizers=True)}]:
        callbacks.append(lbann_pb2.Callback(**dic))

    model = lbann_pb2.Model(
        data_layout = "data_parallel",
        mini_batch_size = 256,
        block_size = 256,
        num_epochs = 10,
        num_parallel_readers = 0,
        procs_per_model = 0,

        objective_function = objective,
        metric = metrics,
        callback = callbacks,
        layer = layers
    )

    with open(outputPath, "w") as f:
        f.write(txtf.MessageToString(lbann_pb2.LbannPB(model=model)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an ONNX model to a LBANN model",
                                     epilog="Usage: onnx2lbann.py model.onnx output.prototext")
    parser.add_argument("onnx_path", type=str,
                        help="Path to an ONNX model")
    parser.add_argument("lbann_path", type=str,
                        help="Path to a LBANN model in .prototext")

    args = parser.parse_args()
    onnxPath = args.onnx_path
    lbannPath = args.lbann_path
    convertAndSave(onnxPath, lbannPath)
