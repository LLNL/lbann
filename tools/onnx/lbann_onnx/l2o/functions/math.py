import onnx
import numpy as np
import lbann_onnx

def parse_relu(lp, inputShapes):
    return {"op": "Relu"}

def parse_leaky_relu(lp, inputShapes):
    return {"op": "LeakyRelu"}

def parse_sigmoid(lp, inputShapes):
    return {"op": "Sigmoid"}

def parse_softmax(lp, inputShapes):
    return {"op": "Softmax"}

def parse_exp(lp, inputShapes):
    return {"op": "Exp"}

def parse_square(lp, inputShapes):
    return {"op": "Identity"} # TODO: this is a dummy operation to perform correct infer_shape

def parse_rsqrt(lp, inputShapes):
    return {"op": "Identity"} # TODO: this is a dummy operation to perform correct infer_shape

def parse_add(lp, inputShapes):
    return {"op": "Sum"}

def parse_sum(lp, inputShapes):
    return {"op": "Sum"}

def parse_hadamard(lp, inputShapes):
    return {"op": "Mul"}

def parse_abs(lp, inputShapes):
    return {"op": "Abs"}

def parse_weighted_sum(lp, inputShapes):
    factors = list(map(float, lp.scaling_factors.split(" ")))
     # TODO: support any weighted_sum
    if factors == [1, 1]:
        return {"op": "Add"}
    elif factors == [1, -1]:
        return {"op": "Sub"}
    elif factors == [0.5, 0.5]:
        return {"op": "Mean"}

    # TODO: this is a dummy operation to perform correct infer_shape
    return {"op": "Sum", "attrs": {"lbannWightedSumFactors": factors}}

def parse_constant(lp, inputShapes):
    shape = list(map(int, lp.num_neurons.split(" ")))
    return {"op": "Constant",
            "attrs": {"value": onnx.helper.make_tensor(name='constant_{}'.format(hash(str(lp.value))),
                                                       data_type=lbann_onnx.ELEM_TYPE,
                                                       dims=shape,
                                                       vals=np.full(shape, float(lp.value)))}}
