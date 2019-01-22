import onnx
import numpy as np

import lbann_onnx
from lbann_onnx.l2o.layers import LbannLayerParser

class LbannLayerParser_relu(LbannLayerParser):
    def parse(self):
        self.appendOperator("Relu")

class LbannLayerParser_leaky_relu(LbannLayerParser):
    def parse(self):
        self.appendOperator("LeakyRelu")

class LbannLayerParser_sigmoid(LbannLayerParser):
    def parse(self):
        self.appendOperator("Sigmoid")

class LbannLayerParser_tanh(LbannLayerParser):
    def parse(self):
        self.appendOperator("Tanh")

class LbannLayerParser_softmax(LbannLayerParser):
    def parse(self):
        self.appendOperator("Softmax")

class LbannLayerParser_exp(LbannLayerParser):
    def parse(self):
        self.appendOperator("Exp")

class LbannLayerParser_add(LbannLayerParser):
    def parse(self):
        self.appendOperator("Sum")

class LbannLayerParser_sum(LbannLayerParser):
    def parse(self):
        self.appendOperator("Sum")

class LbannLayerParser_hadamard(LbannLayerParser):
    def parse(self):
        self.appendOperator("Mul")

class LbannLayerParser_abs(LbannLayerParser):
    def parse(self):
        self.appendOperator("Abs")

class LbannLayerParser_weighted_sum(LbannLayerParser):
    def parse(self):
        params = self.l.weighted_sum
        factors = list(map(float, params.scaling_factors.split(" ")))
        # TODO: support any weighted_sum
        if factors == [1, 1]:
            self.appendOperator("Add")

        elif factors == [1, -1]:
            self.appendOperator("Sub")

        elif factors == [0.5, 0.5]:
            self.appendOperator("Mean")

        else:
            # TODO: this is a dummy operation to perform correct infer_shape
            self.appendOperator("Sum", attrs={"lbannWightedSumFactors": factors})

class LbannLayerParser_constant(LbannLayerParser):
    def parse(self):
        params = self.l.constant
        shape = list(map(int, params.num_neurons.split(" ")))

        self.appendOperator("Constant",
                            attrs={"value": numpy_heler.from_array(np.full(shape,
                                                                           float(params.value),
                                                                           dtype=lbann_onnx.ELEM_TYPE_NP))})

# Dummy parsers

class LbannLayerParser_square(LbannLayerParser):
    def parse(self):
        self.appendOperator("Identity") # TODO: this is a dummy operation to perform correct infer_shape

class LbannLayerParser_rsqrt(LbannLayerParser):
    def parse(self):
        self.appendOperator("Identity") # TODO: this is a dummy operation to perform correct infer_shape
