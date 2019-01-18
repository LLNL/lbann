from lbann_onnx.l2o.layers import LbannLayerParser

class LbannLayerParser_cross_entropy(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannCrossEntropy")

class LbannLayerParser_binary_cross_entropy(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannBinaryCrossEntropy")

class LbannLayerParser_sigmoid_binary_cross_entropy(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannSigmoidBinaryCrossEntropy")

class LbannLayerParser_categorical_accuracy(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannCategoricalAccuracy")

class LbannLayerParser_top_k_categorical_accuracy(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannTopKCategoricalAccuracy")

class LbannLayerParser_mean_absolute_error(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannMeanAbsoluteError")

class LbannLayerParser_mean_squared_error(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannMeanSquaredError")

class LbannLayerParser_l2_norm2(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannL2Norm2")
