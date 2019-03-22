from lbann.onnx.o2l.layers import OnnxLayerParser
from lbann.onnx.parserDescriptor import parserDescriptor
from lbann import lbann_pb2

@parserDescriptor(["local_response_normalization"])
class parse_LRN(OnnxLayerParser):
    def parse(self):
        local_response_normalization = lbann_pb2.LocalResponseNormalization(
            lrn_alpha = self.getNodeAttribute("alpha"),
            lrn_beta = self.getNodeAttribute("beta"),
            lrn_k = self.getNodeAttribute("bias"),
            window_width = self.getNodeAttribute("size"),
        )
        return {"local_response_normalization": local_response_normalization}

@parserDescriptor(["batch_normalization"])
class parse_BatchNormalization(OnnxLayerParser):
    def parse(self):
        batch_normalization = lbann_pb2.BatchNormalization(
            epsilon = self.getNodeAttribute("epsilon", 1e-5),
            decay = self.getNodeAttribute("momentum", 0.9),
        )
        return {"batch_normalization": batch_normalization}

@parserDescriptor(["dropout"])
class parse_Dropout(OnnxLayerParser):
    def parse(self):
        return {"dropout": lbann_pb2.Dropout(keep_prob = 1.0-self.getNodeAttribute("ratio"))}
