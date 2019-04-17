import onnx
import numpy as np

import lbann.onnx
from lbann.onnx.parserDescriptor import parserDescriptor
from lbann.onnx.l2o.layers import LbannLayerParser

@parserDescriptor(["Relu"])
class LbannLayerParser_relu(LbannLayerParser):
    def parse(self):
        self.appendOperator("Relu")

@parserDescriptor(["LeakyRelu"])
class LbannLayerParser_leaky_relu(LbannLayerParser):
    def parse(self):
        self.appendOperator("LeakyRelu")

@parserDescriptor(["Sigmoid"])
class LbannLayerParser_sigmoid(LbannLayerParser):
    def parse(self):
        self.appendOperator("Sigmoid")

@parserDescriptor(["Tanh"])
class LbannLayerParser_tanh(LbannLayerParser):
    def parse(self):
        self.appendOperator("Tanh")

@parserDescriptor(["Softmax"])
class LbannLayerParser_softmax(LbannLayerParser):
    def parse(self):
        self.appendOperator("Softmax")

@parserDescriptor(["Exp"])
class LbannLayerParser_exp(LbannLayerParser):
    def parse(self):
        self.appendOperator("Exp")

@parserDescriptor(["Sum"])
class LbannLayerParser_add(LbannLayerParser):
    def parse(self):
        self.appendOperator("Sum")

@parserDescriptor(["Sum"])
class LbannLayerParser_sum(LbannLayerParser):
    def parse(self):
        self.appendOperator("Sum")

@parserDescriptor(["Mul"])
class LbannLayerParser_hadamard(LbannLayerParser):
    def parse(self):
        self.appendOperator("Mul")

@parserDescriptor(["Abs"])
class LbannLayerParser_abs(LbannLayerParser):
    def parse(self):
        self.appendOperator("Abs")

@parserDescriptor(stub=True)
class LbannLayerParser_weighted_sum(LbannLayerParser):
    def parse(self):
        params = self.l.weighted_sum
        factors = list(map(float, params.scaling_factors.split(" ")))
        if factors == [1, 1]:
            self.appendOperator("Add")

        elif factors == [1, -1]:
            self.appendOperator("Sub")

        elif factors == [0.5, 0.5]:
            self.appendOperator("Mean")

        else:
            raise NotImplementedError()

@parserDescriptor(["Constant"])
class LbannLayerParser_constant(LbannLayerParser):
    def parse(self):
        params = self.l.constant
        shape = list(map(int, params.num_neurons.split(" ")))

        self.appendOperator("Constant",
                            attrs={"value": numpy_heler.from_array(np.full(shape,
                                                                           float(params.value),
                                                                           dtype=lbann.onnx.ELEM_TYPE_NP))})

# Dummy parsers
@parserDescriptor(stub=True)
class LbannLayerParser_square(LbannLayerParser):
    def parse(self):
        raise NotImplementedError()

@parserDescriptor(stub=True)
class LbannLayerParser_rsqrt(LbannLayerParser):
    def parse(self):
        raise NotImplementedError()
