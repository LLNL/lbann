"""
A test suite to check whether lbann.onnx can convert typical ONNX operators to LBANN layers correctly.

For each test case, this script
1. converts a given ONNX operator into a LBANN layer, and
2. compares all the fields of the converted operator and a prepared eqivalent LBANN layer.

If a converted LBANN layer has non-empty "*_i" fields, this script ignores them but compares the vector version of them.
"""

import unittest
import onnx
import re

import lbann.onnx
import lbann.onnx.o2l
from lbann.onnx.util import lbannList2List
from lbann.onnx.tests.util import getLbannVectorField
import lbann

def makeFloatTensorVI(name, shape):
    return onnx.helper.make_tensor_value_info(
        name=name,
        elem_type=lbann.onnx.ELEM_TYPE,
        shape=shape
    )

def makeOnnxModel(node, inputs):
    inputs = list(map(lambda x: makeFloatTensorVI(*x), inputs.items()))
    return onnx.helper.make_model(onnx.helper.make_graph(
        [node],
        name="",
        inputs=inputs,
        outputs=[])
    )

def convertOnnxNode(node, inputs, params):
    o = makeOnnxModel(node, {**inputs, **params})
    layers = lbann.onnx.o2l.onnxToLbannLayers(
        o,
        inputs.keys(),
        dict(zip(inputs.keys(), inputs.keys()))
    )
    return layers[-1]


class TestOnnx2LbannLayer(unittest.TestCase):
    def _assertFields(self, lbannFields, onnxFields):
        hasVectors = False
        if "has_vectors" in lbannFields.get_field_names():
            assert hasattr(onnxFields, "has_vectors")
            hasVectors = True

        for fieldName in lbannFields.get_field_names():
            if fieldName == "has_vectors":
                continue

            if re.compile("_i$").search(fieldName):
                continue

            lbannField = getattr(lbannFields, fieldName)
            onnxField  = getattr(onnxFields,  fieldName)

            if hasVectors and "{}_i".format(fieldName) in lbannFields.get_field_names():
                lbannField = lbannList2List(getLbannVectorField(lbannFields, fieldName))
                onnxField = lbannList2List(getLbannVectorField(onnxFields, fieldName))
                if len(lbannField) < len(onnxField):
                    assert len(lbannField) == 1
                    lbannField *= len(onnxField)

                if len(onnxField) < len(lbannField):
                    assert len(onnxField) == 1
                    onnxField *= len(lbannField)

            if lbannField is None:
                continue

            if hasVectors and re.compile("_i$").search(fieldName):
                continue

            assertFunc = self.assertEqual
            if isinstance(lbannField, float) and isinstance(onnxField, float):
                assertFunc = self.assertAlmostEqual

            assertFunc(
                lbannField,
                onnxField,
                msg=fieldName
            )

    def _test_o2l_layer_Gemm(self, hasBias):
        M, N, K = (100, 200, 300)

        lbannFC = lbann.FullyConnected(
            lbann.Input(),
            num_neurons=N,
            has_bias=hasBias
        )

        inputShapes = {"x": [M,K]}
        paramShapes = {"W": [N,K]}
        if hasBias:
            paramShapes["b"] = [N]

        node = onnx.helper.make_node(
            "Gemm",
            inputs=["x","W"] + (["b"] if hasBias else []),
            outputs=["y"],
            transB=1
        )
        onnxFC = convertOnnxNode(
            node,
            inputShapes,
            paramShapes
        ).fully_connected

        self._assertFields(lbannFC, onnxFC)

    def test_o2l_layer_Gemm_bias(self):
        self._test_o2l_layer_Gemm(hasBias=True)

    def test_o2l_layer_Gemm_no_bias(self):
        self._test_o2l_layer_Gemm(hasBias=False)

    def _test_o2l_layer_Conv(self, numDims, hasBias):
        N, C_in, H = (256, 3, 224)
        C_out = 64
        K, P, S, D = (3, 1, 1, 1)
        G = 1

        lbannConv = lbann.Convolution(
            lbann.Input(),
            num_dims=numDims,
            num_output_channels=C_out,
            has_vectors=False,
            conv_dims_i=K,
            conv_pads_i=P,
            conv_strides_i=S,
            conv_dilations_i=D,
            num_groups=G,
            has_bias=hasBias
        )

        inputShapes = {"x": [N, C_in] + [H]*numDims}
        paramShapes = {"W": [C_out, C_in] + [K]*numDims}
        if hasBias:
            paramShapes["b"] = [C_out]

        node = onnx.helper.make_node(
            "Conv",
            inputs=["x","W"] + (["b"] if hasBias else []),
            outputs=["y"],
            kernel_shape=[K]*numDims,
            pads=[P]*(numDims*2),
            strides=[S]*numDims,
            dilations=[D]*numDims,
            group=G
        )
        onnxConv = convertOnnxNode(
            node,
            inputShapes,
            paramShapes
        ).convolution

        self._assertFields(lbannConv, onnxConv)

    def test_o2l_layer_Conv_bias(self):
        self._test_o2l_layer_Conv(numDims=2, hasBias=True)

    def test_o2l_layer_Conv_no_bias(self):
        self._test_o2l_layer_Conv(numDims=2, hasBias=False)

    def test_o2l_layer_Conv_3D_bias(self):
        self._test_o2l_layer_Conv(numDims=3, hasBias=True)

    def test_o2l_layer_Conv_3D_no_bias(self):
        self._test_o2l_layer_Conv(numDims=3, hasBias=False)

    def _test_o2l_layer_Pool(self, numDims, poolMode, onnxOp):
        N, C, H = (256, 3, 224)
        K, P, S = (3, 1, 1)

        lbannPooling = lbann.Pooling(
            lbann.Input(),
            num_dims=numDims,
            has_vectors=False,
            pool_dims_i=K,
            pool_pads_i=P,
            pool_strides_i=S,
            pool_mode=poolMode
        )

        inputShapes = {"x": [N, C] + [H]*numDims}

        node = onnx.helper.make_node(
            onnxOp,
            inputs=["x"],
            outputs=["y"],
            kernel_shape=[K]*numDims,
            pads=[P]*(numDims*2),
            strides=[S]*numDims,
        )
        onnxPooling = convertOnnxNode(
            node,
            inputShapes,
            {}
        ).pooling

        self._assertFields(lbannPooling, onnxPooling)

    def test_o2l_layer_MaxPool(self):
        self._test_o2l_layer_Pool(numDims=2, poolMode="max", onnxOp="MaxPool")

    def test_o2l_layer_AveragePool(self):
        self._test_o2l_layer_Pool(numDims=2, poolMode="average", onnxOp="AveragePool")

    def test_o2l_layer_MaxPool_3D(self):
        self._test_o2l_layer_Pool(numDims=3, poolMode="max", onnxOp="MaxPool")

    def test_o2l_layer_AveragePool_3D(self):
        self._test_o2l_layer_Pool(numDims=3, poolMode="average", onnxOp="AveragePool")

    def test_o2l_layer_BatchNormalization(self):
        N, C, H, W = (100,200,300,400)
        decay = 0.95
        epsilon = 1e-6

        lbannBN = lbann.BatchNormalization(
            lbann.Input(),
            decay=decay, epsilon=epsilon,
        )

        inputShapes = {"x": [N,C,H,W]}
        paramShapes = {"scale": [C],
                       "B"    : [C],
                       "mean" : [C],
                       "var"  : [C]}

        node = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["x", "scale", "B", "mean", "var"],
            outputs=["y"],
            epsilon=epsilon,
            momentum=decay
        )
        onnxBN = convertOnnxNode(
            node,
            inputShapes,
            paramShapes
        ).batch_normalization

        self._assertFields(lbannBN, onnxBN)

    def test_o2l_layer_Relu(self):
        N, C, H, W = (100,200,300,400)

        lbannRelu = lbann.Relu(
            lbann.Input(),
        )

        node = onnx.helper.make_node(
            "Relu",
            inputs=["x"],
            outputs=["y"],
        )
        onnxRelu = convertOnnxNode(
            node,
            {"x": [N,C,H,W]},
            {}
        ).relu

        self._assertFields(lbannRelu, onnxRelu)


if __name__ == "__main__":
    unittest.main()
