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
Specific PyTorch Module converters for the LBANN PyTorch frontend.
"""
import functools
import lbann
from lbann.torch.converters import (register_module,
                                    register_module_weight_converter,
                                    register_opaque_shape_inference)
import torch
import torch.nn as nn


def convnd_layer(c, dims: int, args, kwargs):
    return lbann.Convolution(*args,
                             **kwargs,
                             num_dims=dims,
                             out_channels=c.out_channels,
                             kernel_size=c.kernel_size,
                             padding=c.padding,
                             stride=c.stride,
                             dilation=c.dilation,
                             has_bias=c.bias is not None,
                             groups=c.groups)


def deconvnd_layer(c: nn.ConvTranspose2d, dims: int, args, kwargs):
    return lbann.Deconvolution(*args,
                               **kwargs,
                               num_dims=dims,
                               out_channels=c.out_channels,
                               kernel_size=c.kernel_size,
                               padding=c.padding,
                               output_padding=c.output_padding,
                               stride=c.stride,
                               dilation=c.dilation,
                               has_bias=c.bias is not None,
                               groups=c.groups)


@register_module(nn.Conv1d)
def conv1d(c: nn.Conv1d, *args, **kwargs):
    return convnd_layer(c, 1, args, kwargs)


@register_module(nn.Conv2d)
def conv2d(c: nn.Conv2d, *args, **kwargs):
    return convnd_layer(c, 2, args, kwargs)


@register_module(nn.Conv3d)
def conv3d(c: nn.Conv3d, *args, **kwargs):
    return convnd_layer(c, 3, args, kwargs)


@register_module(nn.ConvTranspose1d)
def deconv1d(c: nn.ConvTranspose1d, *args, **kwargs):
    return deconvnd_layer(c, 1, args, kwargs)


@register_module(nn.ConvTranspose2d)
def deconv2d(c: nn.ConvTranspose2d, *args, **kwargs):
    return deconvnd_layer(c, 2, args, kwargs)


@register_module(nn.ConvTranspose3d)
def deconv3d(c: nn.ConvTranspose3d, *args, **kwargs):
    return deconvnd_layer(c, 3, args, kwargs)


@register_module(nn.LeakyReLU)
def leaky_relu_impl(origmod: nn.LeakyReLU, *args, **kwargs):
    return lbann.LeakyRelu(*args,
                           **kwargs,
                           negative_slope=origmod.negative_slope)


@register_module(nn.AvgPool2d)
def avgpool2d_impl(origmod: nn.AvgPool2d, *args, **kwargs):
    return lbann.Pooling(*args,
                         **kwargs,
                         pool_mode='average',
                         num_dims=2,
                         has_vectors=False,
                         pool_dims_i=origmod.kernel_size,
                         pool_pads_i=origmod.padding,
                         pool_strides_i=origmod.stride)


@register_module(nn.AvgPool3d)
def avgpool3d_impl(origmod: nn.AvgPool3d, *args, **kwargs):
    return lbann.Pooling(*args,
                         **kwargs,
                         pool_mode='average',
                         num_dims=3,
                         has_vectors=False,
                         pool_dims_i=origmod.kernel_size,
                         pool_pads_i=origmod.padding,
                         pool_strides_i=origmod.stride)


@register_module(nn.Dropout)
def dropout_impl(mod: nn.Dropout, *args, **kwargs):
    return lbann.Dropout(*args, **kwargs, keep_prob=(1 - mod.p))


@register_module(nn.Linear)
def linear_impl(origmod: nn.Linear, *args, **kwargs):
    return lbann.FullyConnected(*args,
                                **kwargs,
                                num_neurons=origmod.out_features,
                                has_bias=origmod.bias is not None,
                                transpose=True)


@register_module(nn.Flatten)
def flatten_impl(mod: nn.Flatten, x, **kwargs):
    if mod.start_dim == 0 or mod.end_dim == 0:
        raise ValueError('Cannot flatten batch dimension in LBANN')

    shp = x.shape
    n = len(shp)
    start_dim = mod.start_dim if mod.start_dim > 0 else n + mod.start_dim
    end_dim = mod.end_dim if mod.end_dim >= 0 else n + mod.end_dim + 1

    new_shape = list(shp[:start_dim]) + [
        functools.reduce(lambda a, b: a * b, shp[start_dim:end_dim], 1)
    ] + list(shp[end_dim:])

    return lbann.Reshape(x, dims=new_shape[1:])


@register_module(nn.ReLU)
def relu_impl(mod: nn.ReLU, *args, **kwargs):
    return lbann.Relu(*args, **kwargs)


@register_module(nn.Identity)
def identity_impl(mod: nn.Identity, x, *args, **kwargs):
    return x  #return lbann.Identity(x)


@register_module(nn.Embedding)
def embedding_impl(mod: nn.Embedding, x, *args, **kwargs):
    return lbann.Embedding(x,
                           num_embeddings=mod.num_embeddings,
                           embedding_dim=mod.embedding_dim,
                           padding_idx=mod.padding_idx)


@register_module(nn.LayerNorm)
def ln_impl(mod: nn.LayerNorm, x, *args, **kwargs):
    return lbann.LayerNorm(x, epsilon=mod.eps)


@register_module([nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d])
def bn_impl(mod: nn.BatchNorm2d, x, *args, **kwargs):
    return lbann.BatchNormalization(x,
                                    decay=1 - mod.momentum,
                                    epsilon=mod.eps,
                                    no_bessel_correction=True)


def make_vector(val, dims):
    if isinstance(val, int):
        return [val] * dims
    return val


def pool_nd_impl(mod: nn.MaxPool2d, x, dims: int):
    assert len(x.shape) == dims + 2  # Batch, channels
    return lbann.Pooling(x,
                         pool_mode='max',
                         num_dims=dims,
                         has_vectors=True,
                         pool_dims=make_vector(mod.kernel_size, dims),
                         pool_pads=make_vector(mod.padding, dims),
                         pool_strides=make_vector(mod.stride, dims))


@register_module(nn.MaxPool1d)
def _impl(mod: nn.MaxPool1d, x):
    return pool_nd_impl(mod, x, 1)


@register_module(nn.MaxPool2d)
def _impl(mod: nn.MaxPool2d, x):
    return pool_nd_impl(mod, x, 2)


@register_module(nn.MaxPool3d)
def _impl(mod: nn.MaxPool3d, x):
    return pool_nd_impl(mod, x, 3)


def aap_nd(x, dims: int):
    assert len(x.shape) == dims + 2  # Batch, channels
    return lbann.ChannelwiseMean(x)


@register_module(nn.AdaptiveAvgPool1d)
def _impl(mod: nn.AdaptiveAvgPool1d, x):
    return aap_nd(x, 1)


@register_module(nn.AdaptiveAvgPool2d)
def _impl(mod: nn.AdaptiveAvgPool2d, x):
    return aap_nd(x, 2)


@register_module(nn.AdaptiveAvgPool3d)
def _impl(mod: nn.AdaptiveAvgPool3d, x):
    return aap_nd(x, 3)


@register_module(nn.Tanh)
def _impl(mod: nn.Tanh, x):
    return lbann.Tanh(x)

#################################################################
# Weight conversion


def _as_weights(scalar_or_array) -> lbann.Weights:
    if hasattr(scalar_or_array, 'shape'):  # Tensor
        if isinstance(scalar_or_array, nn.Parameter):
            scalar_or_array = scalar_or_array.detach().cpu().numpy()
        if isinstance(scalar_or_array, torch.Tensor):
            scalar_or_array = scalar_or_array.detach().cpu().numpy()
        return lbann.Weights(initializer=lbann.ValueInitializer(
            values=scalar_or_array.flat))
    return lbann.Weights(initializer=lbann.ValueInitializer(
        values=[scalar_or_array]))  # Assuming scalar


@register_module_weight_converter([
    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
    nn.ConvTranspose3d
])
@register_module_weight_converter(nn.Linear)
def weights_and_biases(mod: nn.Module,
                       layer: lbann.Layer,
                       transpose: bool = False):
    """
    General nn.Module to lbann.Layer weight converter for modules that have two
    parameters: ``weight`` and ``bias``.

    :param mod: The module to convert parameters from.
    :param layer: The layer to convert weights to.
    :param transpose: If True, transposes the last two dimensions of the weight
                      tensor.
    """
    params = []

    # Obtain and transpose weights
    if hasattr(mod, 'weight') and mod.weight is not None:
        weights_numpy = mod.weight.detach().cpu().numpy()
        if transpose:
            indices = list(range(len(weights_numpy.shape)))
            indices[-2], indices[-1] = indices[-1], indices[-2]
            weights_numpy = weights_numpy.transpose(indices)
        params.append(_as_weights(weights_numpy))

    # Obtain bias
    if hasattr(mod, 'bias') and mod.bias is not None:
        params.append(_as_weights(mod.bias))

    if params:
        layer.weights = params


@register_module_weight_converter(
    [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d])
def batch_norm_weights(mod: nn.BatchNorm2d, layer: lbann.BatchNormalization):
    if mod.affine:  # Scaling and addition
        weights = [mod.weight, mod.bias]
    else:
        weights = [
            torch.ones(mod.num_features).detach().cpu().numpy(),
            torch.zeros(mod.num_features).detach().cpu().numpy(),
        ]

    if mod.track_running_stats:
        # TODO(later): mod.num_batches_tracked is unused
        weights.extend([mod.running_mean, mod.running_var])
    else:
        weights.extend([0, 1])

    layer.weights = [_as_weights(w) for w in weights]


@register_module_weight_converter(nn.Embedding)
def embedding_impl(mod: nn.Embedding, layer: lbann.Embedding):
    layer.weights = [_as_weights(mod.weight)]


#################################################################
# Shape/type inference

# PyTorch Geometric
try:
    from torch_geometric.nn import GCNConv

    @register_opaque_shape_inference(GCNConv)
    def infer(mod: GCNConv, x, edge_index, edge_weight=None):
        new_shape = (x.shape[0], mod.out_channels)
        return x.dtype, new_shape, x.device

    @register_module(GCNConv)
    def _impl(mod: GCNConv, x, edge_index, edge_weight=None):
        raise NotImplementedError

except (ImportError, ModuleNotFoundError):
    # No PyTorch Geometric installation found, skip
    pass
