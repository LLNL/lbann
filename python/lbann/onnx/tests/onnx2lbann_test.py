"""
A test suite to check whether lbann.onnx can convert typical ONNX networks to LBANN networks correctly.

For each test case, this script
1. converts a given ONNX network into a LBANN network, and
2. saves the network to DUMP_DIR (if LBANN_ONNX_DUMP_MODELS is set).

This script does not check produced tensor shapes since the LBANN Python frontend is not capable of
performing shape inference (while the ONNX Python frontend can perform it).

This scirpt assumes that ONNX networks from the ONNX Model Zoo (https://github.com/onnx/models)
are placed under ONNX_MDOEL_ZOO_ROOT.
Run scripts/onnx/download_onnx_model_zoo.sh to download the models.
"""

import os
import unittest

import onnx
import google.protobuf.text_format as txtf

import lbann.onnx.o2l
from lbann import lbann_pb2
from lbann.onnx.util import parseBoolEnvVar, getLbannRoot
from lbann.onnx.tests.util import isModelDumpEnabled, createAndGetDumpedModelsDir

ONNX_MODEL_ZOO_ROOT = "onnx_model_zoo"
SAVE_PROTOTEXT = isModelDumpEnabled()
DUMP_DIR = createAndGetDumpedModelsDir()
DATA_LAYER_NAME = "image"
LABEL_LAYER_NAME = "label"

class TestOnnx2Lbann(unittest.TestCase):

    def _test(self, modelName):
        path = os.path.join(ONNX_MODEL_ZOO_ROOT, modelName, "model.onnx")
        try:
            o = onnx.load(path)
        except FileNotFoundError:
            self.skipTest("ONNX model is not found." \
                          "You may want to download it by running scripts/onnx/download_onnx_model_zoo.py" \
                          ": {}".format(modelName))

        onnxInputLayer, = set(map(lambda x: x.name, o.graph.input)) - set(map(lambda x: x.name, o.graph.initializer))

        layers = lbann.onnx.o2l.onnxToLbannLayers(
            o,
            [DATA_LAYER_NAME, LABEL_LAYER_NAME],
            {DATA_LAYER_NAME: onnxInputLayer},
        )

        model = lbann_pb2.Model(layer = layers)
        pb = lbann_pb2.LbannPB(model=model)

        if SAVE_PROTOTEXT:
            with open(os.path.join(DUMP_DIR, "{}.prototext".format(modelName)), "w") as f:
                f.write(txtf.MessageToString(pb))

    def test_o2l_mnist(self):
        self._test("mnist")

    def test_o2l_alexnet(self):
        self._test("bvlc_alexnet")

    def test_o2l_caffenet(self):
        self._test("bvlc_reference_caffenet")

    def test_o2l_googlenet(self):
        self._test("bvlc_googlenet")

    def test_o2l_vgg19(self):
        self._test("vgg19")

    def test_o2l_resnet50(self):
        self._test("resnet50")

if __name__ == "__main__":
    unittest.main()
