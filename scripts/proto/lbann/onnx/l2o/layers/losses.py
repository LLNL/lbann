import numpy as np

import lbann.onnx
from lbann.onnx.parserDescriptor import parserDescriptor
from lbann.onnx.l2o.layers import LbannLayerParser

# y_p: a prediction
# y: grand trtuh

# 1/|N| * dot(y, 1)
class LbannLossLayerParser(LbannLayerParser):
    def appendLossOperators(self, vec, square=False, minus=False):
        alpha = 1.0/self.inputShapes[0][0] * (-1.0 if minus else 1.0)

        h1,_ = self.appendOperator("ReduceSum" if not square else "ReduceSumSquare",
                                   inputNames=[vec], hiddenOutputCount=1)
        h2 = self.createHiddenTensorName()
        h3,_ = self.appendOperator("Mul", inputNames=[h1, h2], hiddenOutputCount=1)
        self.appendParamWithInit(h2, np.array([alpha], dtype=lbann.onnx.ELEM_TYPE_NP))
        self.appendOperator("Squeeze", inputNames=[h3])

# -y * log(y_p)
@parserDescriptor(arithmetic=True)
class LbannLayerParser_cross_entropy(LbannLossLayerParser):
    def parse(self):
        predicted, truth = self.getLbannInputNames()
        h1,_ = self.appendOperator("Log", inputNames=[predicted], hiddenOutputCount=1)
        h2,_ = self.appendOperator("Mul", inputNames=[truth, h1], hiddenOutputCount=1)
        self.appendLossOperators(h2, minus=True)

# -[y * log(y_p) + (1-y) * log(1-y_p)]
@parserDescriptor(arithmetic=True)
class LbannLayerParser_binary_cross_entropy(LbannLossLayerParser):
    def parse(self):
        one = self.createHiddenTensorName()
        self.appendParamWithInit(one, np.full(self.inputShapes[0], 1.0, dtype=lbann.onnx.ELEM_TYPE_NP))

        predicted, truth = self.getLbannInputNames()
        predictedLog,_ = self.appendOperator("Log", inputNames=[predicted], hiddenOutputCount=1)
        h1,_ = self.appendOperator("Mul", inputNames=[truth, predictedLog], hiddenOutputCount=1)

        one = self.getParamName(1)
        predictedOne,_ = self.appendOperator("Sub", inputNames=[one, predicted], hiddenOutputCount=1)
        truthOne,_ = self.appendOperator("Sub", inputNames=[one, truth], hiddenOutputCount=1)
        predictedOneLog,_ = self.appendOperator("Log", inputNames=[predictedOne], hiddenOutputCount=1)
        h2,_ = self.appendOperator("Mul", inputNames=[truthOne, predictedOneLog], hiddenOutputCount=1)
        h3,_ = self.appendOperator("Add", inputNames=[h1, h2], hiddenOutputCount=1)
        self.appendLossOperators(h3, minus=True)

# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
# max(x, 0) - x * z + log(1 + exp(-abs(x))), where x = logits, z = labels
@parserDescriptor(arithmetic=True)
class LbannLayerParser_sigmoid_binary_cross_entropy(LbannLossLayerParser):
    def parse(self):
        predicted, truth = self.getLbannInputNames()
        predMax,_        = self.appendOperator("Max", inputNames=[predicted], hiddenOutputCount=1)
        predXTruth,_     = self.appendOperator("Mul", inputNames=[predicted, truth], hiddenOutputCount=1)
        predAbs,_        = self.appendOperator("Abs", inputNames=[predicted], hiddenOutputCount=1)
        predAbsNeg,_     = self.appendOperator("Neg", inputNames=[predAbs], hiddenOutputCount=1)
        predAbsNegExp,_  = self.appendOperator("Exp", inputNames=[predAbsNeg], hiddenOutputCount=1)

        one = self.createHiddenTensorName()
        self.appendParamWithInit(one, np.full(self.inputShapes[0], 1.0, dtype=lbann.onnx.ELEM_TYPE_NP))
        predAbsNegExpPOne,_ = self.appendOperator("Add", inputNames=[predAbsNegExp, one], hiddenOutputCount=1)
        predAbsNegExpPOneLog,_ = self.appendOperator("Log", inputNames=[predAbsNegExpPOne], hiddenOutputCount=1)

        h1,_ = self.appendOperator("Sub", inputNames=[predMax, predXTruth], hiddenOutputCount=1)
        h2,_ = self.appendOperator("Add", inputNames=[h1, predAbsNegExpPOneLog], hiddenOutputCount=1)

        self.appendLossOperators(h2)

# y * hardmax(y_p)
@parserDescriptor(arithmetic=True)
class LbannLayerParser_categorical_accuracy(LbannLossLayerParser):
    def parse(self):
        predicted, truth = self.getLbannInputNames()
        h1,_ = self.appendOperator("Hardmax", inputNames=[predicted], hiddenOutputCount=1)
        h2,_ = self.appendOperator("Mul", inputNames=[truth, h1], hiddenOutputCount=1)
        self.appendLossOperators(h2)

@parserDescriptor(stub=True)
class LbannLayerParser_top_k_categorical_accuracy(LbannLayerParser):
    def parse(self):
        raise NotImplementedError()

class LbannMeanLossLayerParser(LbannLossLayerParser):
    def appendMeanLossOperators(self, square=False):
        predicted, truth = self.getLbannInputNames()
        h1,_ = self.appendOperator("Sub", inputNames=[truth, predicted], hiddenOutputCount=1)
        h2,_ = self.appendOperator("Abs", inputNames=[h1], hiddenOutputCount=1)
        self.appendLossOperators(h2, square=square)

@parserDescriptor(arithmetic=True)
class LbannLayerParser_mean_absolute_error(LbannMeanLossLayerParser):
    def parse(self):
        self.appendMeanLossOperators()

@parserDescriptor(arithmetic=True)
class LbannLayerParser_mean_squared_error(LbannMeanLossLayerParser):
    def parse(self):
        self.appendMeanLossOperators(True)

@parserDescriptor(stub=True)
class LbannLayerParser_l2_norm2(LbannLayerParser):
    def parse(self):
        raise NotImplementedError()
