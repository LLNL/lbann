from lbann.onnx.o2l.layers import OnnxLayerParser, OnnxSpatialLayerParser
from lbann.onnx.parserDescriptor import parserDescriptor
from lbann import lbann_pb2

@parserDescriptor(["convolution"])
class parse_Conv(OnnxSpatialLayerParser):
    def parse(self):
        num_dims = len(self.inputShapes[0])-2
        group = self.getNodeAttribute("group", 1)
        convolution = lbann_pb2.Convolution(
            num_output_channels = self.outputShapes[0][1],
            num_groups = group,
            has_bias = (len(self.inputShapes) == 3),
            **self.parse_Spatial(num_dims, "conv", True),
        )
        return {"convolution": convolution}

@parserDescriptor(["fully_connected"])
class parse_Gemm(OnnxLayerParser):
    def parse(self):
        assert self.getNodeAttribute("transA",0) == 0
        assert self.getNodeAttribute("alpha",1.0) == 1.0
        assert (self.getNodeAttribute("beta",1.0) == 1.0 or len(self.inputShapes) < 3)
        fully_connected = lbann_pb2.FullyConnected(
            num_neurons = self.outputShapes[0][1],
            has_bias = (len(self.inputShapes) == 3),
            transpose = self.getNodeAttribute("transB",0) == 1
        )
        return {"fully_connected": fully_connected}

@parserDescriptor(["fully_connected"])
class parse_MatMul(OnnxLayerParser):
    def parse(self):
        fully_connected = lbann_pb2.FullyConnected(
            num_neurons = self.outputShapes[0][1],
            has_bias = False
        )
        return {"fully_connected": fully_connected}
