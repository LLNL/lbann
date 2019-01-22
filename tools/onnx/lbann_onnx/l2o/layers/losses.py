import numpy as np

import lbann_onnx
from lbann_onnx.l2o.layers import LbannLayerParser

# TODO: use keyword arguments for appendOperator

# y_p: a prediction
# y: grand trtuh

# 1/|N| * dot(y, 1)
class LbannLossLayerParser(LbannLayerParser):
    # TODO: override parse itself
    def appendLossOperators(self, vec, square=False, minus=False):
        alpha = 1.0/self.inputShapes[0][0] * (-1.0 if minus else 1.0)

        h1,_ = self.appendOperator("ReduceSum" if not square else "ReduceSumSquare",
                                   {}, 0, [vec], 1)
        # TODO: deprecate paramShapes or createHiddenTensor
        h2 = self.createHiddenTensorName()
        h3,_ = self.appendOperator("Mul", {}, 0, [h1, h2], 1)
        self.appendParamWithInit(h2,
                                 shape=[1],
                                 data=np.array([alpha], dtype=lbann_onnx.ELEM_TYPE_NP))
        self.appendOperator("Squeeze", {}, 0, [h3])

# -y * log(y_p)
class LbannLayerParser_cross_entropy(LbannLossLayerParser):
    def parse(self):
        predicted, truth = self.getLbannInputNames()
        h1,_ = self.appendOperator("Log", {}, 0, [predicted], 1)
        h2,_ = self.appendOperator("Mul", {}, 0, [truth, h1], 1)
        self.appendLossOperators(h2, minus=True)

# -[y * log(y_p) + (1-y) * log(1-y_p)]
class LbannLayerParser_binary_cross_entropy(LbannLossLayerParser):
    def parse(self):
        one = self.createHiddenTensorName()
        self.appendParamWithInit(one,
                                 shape=self.inputShapes[0],
                                 data=np.full(self.inputShapes[0], 1.0, dtype=lbann_onnx.ELEM_TYPE_NP))

        predicted, truth = self.getLbannInputNames()
        predictedLog,_ = self.appendOperator("Log", {}, 0, [predicted], 1)
        h1,_ = self.appendOperator("Mul", {}, 0, [truth, predictedLog], 1)

        one = self.getParamName(1)
        predictedOne,_ = self.appendOperator("Sub", {}, 0, [one, predicted], 1)
        truthOne,_ = self.appendOperator("Sub", {}, 0, [one, truth], 1)
        predictedOneLog,_ = self.appendOperator("Log", {}, 0, [predictedOne], 1)
        h2,_ = self.appendOperator("Mul", {}, 0, [truthOne, predictedOneLog], 1)
        h3,_ = self.appendOperator("Add", {}, 0, [h1, h2], 1)
        self.appendLossOperators(h3, minus=True)

# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
# max(x, 0) - x * z + log(1 + exp(-abs(x))), where x = logits, z = labels
class LbannLayerParser_sigmoid_binary_cross_entropy(LbannLossLayerParser):
    def parse(self):
        predicted, truth = self.getLbannInputNames()
        predMax,_        = self.appendOperator("Max",{}, 0, [predicted], 1)
        predXTruth,_     = self.appendOperator("Mul",{}, 0, [predicted, truth], 1)
        predAbs,_        = self.appendOperator("Abs",{}, 0, [predicted], 1)
        predAbsNeg,_     = self.appendOperator("Neg",{}, 0, [predAbs], 1)
        predAbsNegExp,_  = self.appendOperator("Exp",{}, 0, [predAbsNeg], 1)

        one = self.createHiddenTensorName()
        self.appendParamWithInit(one,
                                 shape=self.inputShapes[0],
                                 data=np.full(self.inputShapes[0], 1.0, dtype=lbann_onnx.ELEM_TYPE_NP))
        predAbsNegExpPOne,_ = self.appendOperator("Add",{}, 0, [predAbsNegExp, one], 1)
        predAbsNegExpPOneLog,_ = self.appendOperator("Log",{}, 0, [predAbsNegExpPOne], 1)

        h1,_ = self.appendOperator("Sub",{}, 0, [predMax, predXTruth], 1)
        h2,_ = self.appendOperator("Add",{}, 0, [h1, predAbsNegExpPOneLog], 1)

        self.appendLossOperators(h2)

# y * hardmax(y_p)
class LbannLayerParser_categorical_accuracy(LbannLossLayerParser):
    def parse(self):
        predicted, truth = self.getLbannInputNames()
        h1,_ = self.appendOperator("Hardmax", {}, 0, [predicted], 1)
        h2,_ = self.appendOperator("Mul", {}, 0, [truth, h1], 1)
        self.appendLossOperators(h2)

# TODO: implement
class LbannLayerParser_top_k_categorical_accuracy(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannTopKCategoricalAccuracy")

class LbannMeanLossLayerParser(LbannLossLayerParser):
    def appendMeanLossOperators(self, square=False):
        predicted, truth = self.getLbannInputNames()
        h1,_ = self.appendOperator("Sub", {}, 0, [truth, predicted], 1)
        h2,_ = self.appendOperator("Abs", {}, 0, [h1], 1)
        self.appendLossOperators(h2, square=square)

class LbannLayerParser_mean_absolute_error(LbannMeanLossLayerParser):
    def parse(self):
        self.appendMeanLossOperators()

class LbannLayerParser_mean_squared_error(LbannMeanLossLayerParser):
    def parse(self):
        self.appendMeanLossOperators(True)

# TODO: implement
class LbannLayerParser_l2_norm2(LbannLayerParser):
    def parse(self):
        self.appendOperator("LbannL2Norm2")
