#!/usr/bin/env python3

"""
Convert a LBANN model to an ONNX model.
Run "./lbann2onnx.py --help" for more details.
"""

import argparse
import re
import onnx
import os

import lbann.onnx.l2o

def parseInputShape(s):
    name, shape = re.compile("^([^=]+)=([0-9,]+)$").search(s).groups()
    return (name, list(map(int, shape.split(","))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a LBANN model to an ONNX model",
                                     epilog="Usage: lbann2onnx.py model_alexnet.prototext output.onnx image=3,224,224 label=1000")
    parser.add_argument("lbann_path", type=str,
                        help="Path to a LBANN model in .prototext")
    parser.add_argument("onnx_path", type=str,
                        help="Path to an ONNX model")
    parser.add_argument("input_shape", type=str, nargs="*",
                        help="Shape(s) of input tensor(s) *without* the mini-batch size in the \"NAME=N1,...,ND\" format.")
    parser.add_argument("--add-value-info", dest="add_value_info", action="store_const",
                        const=True, default=False,
                        help="Embed value_info in the generated ONNX model")

    args = parser.parse_args()
    lbannPath = args.lbann_path
    onnxPath = args.onnx_path
    inputShapes = dict(map(parseInputShape, args.input_shape))
    addValueInfo = args.add_value_info

    model, miniBatchSize = lbann.onnx.l2o.parseLbannModelPB(os.path.expanduser(lbannPath),
                                                 inputShapes,
                                                 addValueInfo=addValueInfo)
    onnx.save(model, os.path.expanduser(onnxPath))
