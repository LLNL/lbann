from lbann_onnx.o2l.layers import OnnxLayerParser
import lbann_pb2

class parse_Relu(OnnxLayerParser):
    def parse(self):
        return {"relu": lbann_pb2.Relu()}

class parse_Softmax(OnnxLayerParser):
    def parse(self):
        return {"softmax": lbann_pb2.Softmax()}

class parse_Concat(OnnxLayerParser):
    def parse(self):
        return {"concatenation": lbann_pb2.Concatenation(concatenation_axis = self.getNodeAttribute("axis"))}

class parse_Sum(OnnxLayerParser):
    def parse(self):
        return {"sum": lbann_pb2.Sum()}

class parse_Add(OnnxLayerParser):
    def parse(self):
        return {"add": lbann_pb2.Add()}
