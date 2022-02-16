import numpy as np

import lbann
import lbann.modules
from lbann.util import str_list

# Mimics torch.matmul in LBANN
def PytorchMatmul(x, x_shape, y, y_shape, return_dims=False):
    if len(x_shape) != len(y_shape):
        raise RuntimeError(
            "Broadcasting not fully implemented, tensors must have same dimension"
        )
    need_reshape = (len(x_shape) > 3) and (len(y_shape) > 3)
    if need_reshape:
        if x_shape[:-2] != y_shape[:-2]:
            raise RuntimeError("The first n-2 dimensions must match")
        new_x_shape = (np.prod(x_shape[:-2]),) + x_shape[-2:]
        x = lbann.Reshape(x, dims=str_list(new_x_shape))

        new_y_shape = (np.prod(y_shape[:-2]),) + y_shape[-2:]
        y = lbann.Reshape(y, dims=str_list(new_y_shape))

    z = lbann.MatMul(x, y)

    z_shape = x_shape[:-1] + (y_shape[-1],)
    if need_reshape:
        z = lbann.Reshape(z, dims=str_list(z_shape))

    if return_dims:
        return z, z_shape
    return z


# Mimics torch.nn.Linear in LBANN
def PytorchLinear(x, input_shape, hidden_size, weights=[], name="", return_dims=False):
    need_reshape = len(input_shape) > 2
    if need_reshape:
        new_in_shape = (np.prod(input_shape[:-1]), input_shape[-1])
        x = lbann.Reshape(x, dims=str_list(new_in_shape))

    if len(input_shape) == 1:
        y = lbann.FullyConnected(x, num_neurons=hidden_size, weights=weights, name=name)
    else:
        y = lbann.ChannelwiseFullyConnected(
            x, output_channel_dims=[hidden_size], weights=weights, name=name
        )

    if need_reshape:
        new_out_shape = input_shape[:-1] + (hidden_size,)
        y = lbann.Reshape(y, dims=str_list(new_out_shape))
    else:
        new_out_shape = (input_shape[0], hidden_size)

    if return_dims:
        return y, new_out_shape
    return y


# Mimics torch.nn.layernorm in LBANN
def PytorchLayerNorm(x, epsilon, input_shape, weights=[], name=""):
    if len(input_shape) > 2:
        x = lbann.Reshape(
            x, dims=str_list([np.prod(input_shape[:-1]), input_shape[-1]])
        )
    x = lbann.InstanceNorm(x, epsilon=epsilon)
    x = lbann.Reshape(x, dims=str_list(input_shape))
    if weights is not []:
        x, new_x_shape = lbann.modules.Permute(x, input_shape, return_dims=True)
        x = lbann.ChannelwiseScaleBias(x, weights=weights)
        x, _ = lbann.modules.Permute(x, new_x_shape, return_dims=True, name=name)

    return x
