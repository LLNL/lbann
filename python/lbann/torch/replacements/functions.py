################################################################################
# Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. <lbann-dev@llnl.gov>
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LLNL/LBANN.
#
# Licensed under the Apache License, Version 2.0 (the "Licensee"); you
# may not use this file except in compliance with the License.  You may
# obtain a copy of the License at:
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the license.
#
################################################################################
"""
Specific function converters for the LBANN PyTorch frontend.
"""

import functools
import lbann
from lbann.torch.converters import register_function
import torch.nn as nn
import warnings


@register_function('aten.full.default')
def _full(shape, value, **kwargs):
    return lbann.Constant(num_neurons=shape[-1], value=value)


@register_function('aten.view.default')
def reshape(x, new_shape, **kwargs):
    return lbann.Reshape(x, dims=new_shape[1:])


@register_function('aten.slice.Tensor')
def _slice(x, dim, start, end):
    return lbann.Slice(x, axis=dim, slice_points=[start, end])


@register_function('aten.unsqueeze.default')
def unsqueeze(x, dim):
    new_shape = list(x.shape[0:dim]) + [1] + list(x.shape[dim:])
    return lbann.Reshape(x, dims=new_shape[1:])


@register_function('<built-in function add>')  # operator.add
@register_function('<built-in function iadd>')  # operator.iadd
@register_function('aten.add.Tensor')
def add(x, y, **kwargs):
    if isinstance(y, (int, float)):
        return lbann.AddConstant(x, constant=y)
    if isinstance(x, (int, float)):
        return lbann.AddConstant(y, constant=x)

    return lbann.Add(x, y)


@register_function('<built-in function sub>')  # operator.sub
@register_function('<built-in function isub>')  # operator.isub
@register_function('aten.sub.Tensor')
def sub(x, y, **kwargs):
    if isinstance(y, (int, float)):
        return lbann.SubtractConstant(x, constant=y)
    if isinstance(x, (int, float)):
        return lbann.ConstantSubtract(y, constant=x)

    return lbann.Subtract(x, y)


@register_function('<built-in function mul>')  # operator.mul
@register_function('<built-in function imul>')  # operator.imul
@register_function('aten.mul.Tensor')
def mul(x, y, **kwargs):
    if isinstance(y, (int, float)):
        return lbann.Scale(x, constant=y)
    if isinstance(x, (int, float)):
        return lbann.Scale(y, constant=x)

    return lbann.Multiply(x, y)


@register_function('<built-in function truediv>')  # operator.truediv
@register_function('<built-in function itruediv>')  # operator.itruediv
@register_function('aten.div.Tensor')
def div(x, y, **kwargs):
    if isinstance(y, (int, float)):
        return lbann.Scale(x, constant=1 / y)
    if isinstance(x, (int, float)):
        return lbann.Scale(lbann.Reciprocal(y), constant=x)

    return lbann.Divide(x, y)


@register_function('torch.sin')
def _impl(x, **kwargs):
    return lbann.Sin(x)


@register_function('torch.cos')
def _impl(x, **kwargs):
    return lbann.Cos(x)


@register_function('torch.flatten')
def flatten_impl(x, start_dim=1, end_dim=-1):
    if start_dim == 0 or end_dim == 0:
        raise ValueError('Cannot flatten batch dimension in LBANN')

    shp = x.shape
    n = len(shp)
    start_dim = start_dim if start_dim > 0 else n + start_dim
    end_dim = end_dim if end_dim >= 0 else n + end_dim + 1

    new_shape = list(shp[:start_dim]) + [
        functools.reduce(lambda a, b: a * b, shp[start_dim:end_dim], 1)
    ] + list(shp[end_dim:])

    return lbann.Reshape(x, dims=new_shape[1:])


@register_function('torch.nn.functional.batch_norm')
def batchnorm_impl(x,
                   running_mean,
                   running_var,
                   w,
                   b=None,
                   training=False,
                   momentum=0.1,
                   eps=1e-5):
    return lbann.BatchNormalization(x, decay=1 - momentum, epsilon=eps)


@register_function('torch.nn.functional.softmax')
def softmax_impl(x, dim=None, **kwargs):
    return lbann.Softmax(x)


@register_function('prims.convert_element_type.default')
def conv(x, dtype, **kwargs):
    warnings.warn(f'Casting to {dtype} will not be converted')
    return x


@register_function('aten.permute.default')
def permute(x, dims, **kwargs):
    if dims[0] != 0:
        raise NotImplementedError('Batch dimension cannot be permuted when '
                                  'converting to LBANN')
    new_dims = [d - 1 for d in dims[1:]]
    return lbann.TensorPermute(x, axes=new_dims)


@register_function('aten.clone.default')
def clone(x, **kwargs):
    return lbann.Identity(x)


@register_function('aten.clamp_min.default')
def clamp_min(x, clampval):
    return lbann.Clamp(x, min=clampval)


@register_function('aten.clamp_max.default')
def clamp_max(x, clampval):
    return lbann.Clamp(x, max=clampval)


@register_function('aten.tanh.default')
def tanh(x):
    return lbann.Tanh(x)


@register_function('aten.mean.dim')
def mean(x, dims, keepdim=False, **kwargs):
    return lbann.ChannelwiseMean(x)


@register_function('torch._C._nn.linear')
def linear(x, weight, bias=None, **kwargs):
    return lbann.FullyConnected(
        x,
        num_neurons=weight.shape[0],
        has_bias=(True if (bias is not None and bias.shape) else False))


@register_function('torch.nn.functional.relu')
def relu(x):
    return lbann.Relu(x)


@register_function('torch.nn.functional.max_pool2d')
def mp2d(x, kernel_size, strides=None, padding=0, **kwargs):
    assert len(x.shape) == 4  # Batch, channels
    return lbann.Pooling(x,
                         pool_mode='max',
                         num_dims=2,
                         has_vectors=True,
                         pool_dims=kernel_size,
                         pool_pads=padding or [0, 0],
                         pool_strides=strides or [1, 1])


@register_function('torch.cat')
def cat(args, dim=1):
    return lbann.Concatenation(*args, axis=dim)


@register_function('torch.tile')
def tile(x, dims):
    ndims = len(x.shape) - 1  # Remove batch dimension
    if len(dims) > ndims:
        dims = dims[1:]

    if len(dims) < ndims:  # Pad with ones
        dims = [1] * (ndims - len(dims)) + dims

    new_shape = [s * d for s, d in zip(x.shape[1:], dims)]
    return lbann.Tessellate(x, dims=new_shape)


@register_function('aten.expand.default')
def expand(x, dims):
    if tuple(dims) == tuple(x.shape):
        return x
    raise NotImplementedError('Expand not implemented for differing shapes')


@register_function('aten.bmm.default')
def bmm(x, y, **kwargs):
    return lbann.MatMul(x, y)


@register_function('aten.erf.default')
def erf(x):
    return lbann.Erf(x)


@register_function('aten.exp.default')
def exp(x):
    return lbann.Exp(x)


@register_function('aten.select.int')
def select_int(x, dim: int, index: int):
    return lbann.Slice(x, axis=dim, slice_points=[index, index])


@register_function('aten.amax.default')
def amax(x, dims, keepdim=False):
    # Dimension-wise max
    if len(dims) == 1 and x.shape[dims[0]] == 1 and keepdim:
        return x
    raise NotImplementedError('Dimension-wise max not implemented')


@register_function('torch.full')
def full(size, fill_value, **kwargs):
    return lbann.Constant(value=fill_value, num_neurons=size)


@register_function('torch.addmm')
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    rhs = lbann.Scale(lbann.MatMul(mat1, mat2), constant=alpha)
    return lbann.Add(lbann.Scale(input, constant=beta), rhs)
