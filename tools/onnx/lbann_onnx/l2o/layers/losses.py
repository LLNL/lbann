from lbann_onnx.l2o.layers import LbannLayerParser

class LbannLayerParser_cross_entropy(LbannLayerParser):
    def parse(self):
        return {"op": "LbannCrossEntropy"}

class LbannLayerParser_binary_cross_entropy(LbannLayerParser):
    def parse(self):
        return {"op": "LbannBinaryCrossEntropy"}

class LbannLayerParser_sigmoid_binary_cross_entropy(LbannLayerParser):
    def parse(self):
        return {"op": "LbannSigmoidBinaryCrossEntropy"}

class LbannLayerParser_categorical_accuracy(LbannLayerParser):
    def parse(self):
        return {"op": "LbannCategoricalAccuracy"}

class LbannLayerParser_top_k_categorical_accuracy(LbannLayerParser):
    def parse(self):
        return {"op": "LbannTopKCategoricalAccuracy"}

class LbannLayerParser_mean_absolute_error(LbannLayerParser):
    def parse(self):
        return {"op": "LbannMeanAbsoluteError"}

class LbannLayerParser_mean_squared_error(LbannLayerParser):
    def parse(self):
        return {"op": "LbannMeanSquaredError"}

class LbannLayerParser_l2_norm2(LbannLayerParser):
    def parse(self):
        return {"op": "LbannL2Norm2"}
