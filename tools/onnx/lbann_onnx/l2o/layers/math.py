import onnx
import numpy as np

import lbann_onnx
from lbann_onnx.l2o.layers import LbannLayerParser

class LbannLayerParser_relu(LbannLayerParser):
    def parse(self):
        return {"op": "Relu"}

class LbannLayerParser_leaky_relu(LbannLayerParser):
    def parse(self):
        return {"op": "LeakyRelu"}

class LbannLayerParser_sigmoid(LbannLayerParser):
    def parse(self):
        return {"op": "Sigmoid"}

class LbannLayerParser_tanh(LbannLayerParser):
    def parse(self):
        return {"op": "Tanh"}

class LbannLayerParser_softmax(LbannLayerParser):
    def parse(self):
        return {"op": "Softmax"}

class LbannLayerParser_exp(LbannLayerParser):
    def parse(self):
        return {"op": "Exp"}

class LbannLayerParser_add(LbannLayerParser):
    def parse(self):
        return {"op": "Sum"}

class LbannLayerParser_sum(LbannLayerParser):
    def parse(self):
        return {"op": "Sum"}

class LbannLayerParser_hadamard(LbannLayerParser):
    def parse(self):
        return {"op": "Mul"}

class LbannLayerParser_abs(LbannLayerParser):
    def parse(self):
        return {"op": "Abs"}

class LbannLayerParser_weighted_sum(LbannLayerParser):
    def parse(self):
        params = self.l.weighted_sum
        factors = list(map(float, params.scaling_factors.split(" ")))
         # TODO: support any weighted_sum
        if factors == [1, 1]:
            return {"op": "Add"}
        elif factors == [1, -1]:
            return {"op": "Sub"}
        elif factors == [0.5, 0.5]:
            return {"op": "Mean"}

        # TODO: this is a dummy operation to perform correct infer_shape
        return {"op": "Sum", "attrs": {"lbannWightedSumFactors": factors}}

class LbannLayerParser_constant(LbannLayerParser):
    def parse(self):
        params = self.l.constant
        shape = list(map(int, params.num_neurons.split(" ")))
        return {"op": "Constant",
                "attrs": {"value": onnx.helper.make_tensor(name='constant_{}'.format(hash(str(params.value))),
                                                           data_type=lbann_onnx.ELEM_TYPE,
                                                           dims=shape,
                                                           vals=np.full(shape, float(params.value)))}}

# Dummy parsers

class LbannLayerParser_square(LbannLayerParser):
    def parse(self):
        return {"op": "Identity"} # TODO: this is a dummy operation to perform correct infer_shape

class LbannLayerParser_rsqrt(LbannLayerParser):
    def parse(self):
        return {"op": "Identity"} # TODO: this is a dummy operation to perform correct infer_shape
