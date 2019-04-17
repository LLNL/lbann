from lbann.onnx.o2l.layers import OnnxLayerParser
from lbann.onnx.parserDescriptor import parserDescriptor
from lbann import lbann_pb2

@parserDescriptor(["relu"])
class parse_Relu(OnnxLayerParser):
    def parse(self):
        return {"relu": lbann_pb2.Relu()}

@parserDescriptor(["softmax"])
class parse_Softmax(OnnxLayerParser):
    def parse(self):
        return {"softmax": lbann_pb2.Softmax()}

@parserDescriptor(["concatenation"])
class parse_Concat(OnnxLayerParser):
    def parse(self):
        return {"concatenation": lbann_pb2.Concatenation(axis = self.getNodeAttribute("axis"))}

@parserDescriptor(["sum"])
class parse_Sum(OnnxLayerParser):
    def parse(self):
        return {"sum": lbann_pb2.Sum()}

@parserDescriptor(["add"])
class parse_Add(OnnxLayerParser):
    def parse(self):
        return {"add": lbann_pb2.Add()}
