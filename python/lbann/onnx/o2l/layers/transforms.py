from lbann.onnx.o2l.layers import OnnxLayerParser, OnnxSpatialLayerParser
from lbann.onnx.parserDescriptor import parserDescriptor
from lbann.onnx.util import list2LbannList
from lbann import lbann_pb2

@parserDescriptor(["relu"])
class OnnxPoolingLayerParser(OnnxSpatialLayerParser):
    def parse_MaxAveragePool(self, average):
        num_dims = len(self.inputShapes[0])-2
        pooling = lbann_pb2.Pooling(
            pool_mode = "average" if average else "max",
            **self.parse_Spatial(num_dims, "pool", False),
        )
        return {"pooling": pooling}

@parserDescriptor(["pooling"])
class parse_MaxPool(OnnxPoolingLayerParser):
    def parse(self):
        return self.parse_MaxAveragePool(average=False)

@parserDescriptor(["pooling"])
class parse_AveragePool(OnnxPoolingLayerParser):
    def parse(self):
        return self.parse_MaxAveragePool(average=True)

@parserDescriptor(["reshape"])
class parse_Reshape(OnnxLayerParser):
    def parse(self):
        reshape = lbann_pb2.Reshape(
            num_dims = len(self.inputShapes[1]),
            dims = list2LbannList(self.inits[1])
        )
        return {"reshape": reshape}
