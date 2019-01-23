import os
import unittest

import onnx
import google.protobuf.text_format as txtf

import lbann_onnx.o2l
import lbann_pb2
from lbann_onnx import getLbannRoot
from lbann_onnx.util import parseBoolEnvVar
from lbann_onnx.tests.util import isModelDumpEnabled, createAndGetDumpedModelsDir

ONNX_MODEL_ZOO_ROOT = "{}/tools/onnx/onnx_model_zoo".format(getLbannRoot())
SAVE_PROTOTEXT = isModelDumpEnabled()
DUMP_DIR = createAndGetDumpedModelsDir()
DATA_LAYER_NAME = "image"
LABEL_LAYER_NAME = "label"

class TestOnnx2Lbann(unittest.TestCase):
    def _test(self, modelName):
        o = onnx.load(os.path.join(ONNX_MODEL_ZOO_ROOT, modelName, "model.onnx"))
        onnxInputLayer, = set(map(lambda x: x.name, o.graph.input)) - set(map(lambda x: x.name, o.graph.initializer))

        layers = lbann_onnx.o2l.onnxToLbannLayers(
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
