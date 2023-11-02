"""Support for automatic mixed precision."""

from typing import Optional

import functools
import argparse

import lbann


# Define layers that can be converted to FP16, or that must run in FP32.
# Layers not specified will have their datatype set based on their
# inputs.
# If a layer has an explicit datatype provided, it will be respected.
FP16_LAYERS = frozenset([
    lbann.Convolution,
    lbann.Deconvolution,
    lbann.FullyConnected,
    lbann.ChannelwiseFullyConnected,
    lbann.Embedding,
    lbann.MatMul
])

FP32_LAYERS = frozenset([
    lbann.Input,
    lbann.CrossEntropy,
    lbann.MeanSquaredError,
    lbann.MeanAbsoluteError,
    lbann.L1Norm,
    lbann.L2Norm2,
    lbann.Softmax,
    lbann.LogSoftmax,
    lbann.ChannelwiseSoftmax,
    lbann.LayerNorm
])


def num_weights_by_bias(layer: lbann.Layer, bias_field: str = 'has_bias') -> int:
    """Helper for determining the number of weights some layers have.

    Returns 2 if the layer has a bias enabled, 1 otherwise.

    """
    has_bias = getattr(layer, bias_field)
    if has_bias:
        return 2
    return 1

# These define the number of weights to add for a given layer if it
# does not have weights.
NUM_WEIGHTS = {
    lbann.Convolution: num_weights_by_bias,
    lbann.Deconvolution: num_weights_by_bias,
    lbann.FullyConnected: num_weights_by_bias,
    lbann.ChannelwiseFullyConnected: functools.partial(
        num_weights_by_bias, bias_field='bias'),
    lbann.BatchNormalization: lambda _: 4,
    lbann.LayerNorm: lambda _: 2,
}

# These define initializer types for a layer.
WEIGHTS_INITIALIZERS = {
    lbann.Convolution: [lbann.HeNormalInitializer,
                        lambda: lbann.ConstantInitializer(value=0.0)],
    lbann.Deconvolution: [lbann.HeNormalInitializer,
                          lambda: lbann.ConstantInitializer(value=0.0)],
    lbann.FullyConnected: [lbann.HeNormalInitializer,
                           lambda: lbann.ConstantInitializer(value=0.0)],
    lbann.ChannelwiseFullyConnected: [lbann.HeNormalInitializer,
                                      lambda: lbann.ConstantInitializer(value=0.0)],
    lbann.BatchNormalization: [lambda: lbann.ConstantInitializer(value=1.0),
                               lambda: lbann.ConstantInitializer(value=0.0),
                               lambda: lbann.ConstantInitializer(value=0.0),
                               lambda: lbann.ConstantInitializer(value=1.0)],
    lbann.LayerNorm: [lambda: lbann.ConstantInitializer(value=1.0),
                      lambda: lbann.ConstantInitializer(value=0.0)],
}

# These define whether to enable or disable optimizers for specific weights.
# None means to use the default optimizer.
WEIGHTS_OPTIMIZERS = {
    lbann.BatchNormalization: [lambda: None,
                               lambda: None,
                               lbann.NoOptimizer,
                               lbann.NoOptimizer]
}


def add_weights(model: lbann.Model) -> None:
    """Set up weights in the model.

    If a layer that should have weights does not, they will be added in
    FP32 precision. If there are existing weights without a datatype,
    their datatype will be set to FP32.

    """
    for layer in model.layers:
        layer_type = type(layer)
        # Set weights to FP32 if they don't have a datatype set.
        for weight in layer.weights:
            if weight.datatype is None:
                weight.datatype = lbann.DataType.FLOAT
        # If only some weights are present, add the remainder.
        if layer_type in NUM_WEIGHTS:
            num_weights = NUM_WEIGHTS[layer_type](layer)
            initializer_types = WEIGHTS_INITIALIZERS[layer_type]
            has_optimizer = WEIGHTS_OPTIMIZERS.get(layer_type, [])
            # Generate the weights.
            weights = [lbann.Weights(
                name=f'{layer.name}_w{i}',
                datatype=lbann.DataType.FLOAT,
                initializer=initializer_types[i](),
                optimizer=(has_optimizer[i]() if len(has_optimizer) > i else None))
                       for i in range(num_weights)]
            # Drop existing weights.
            weights = weights[len(layer.weights):]
            layer.add_weights(weights)
            model.weights.update(weights)


def get_widest_datatype(datatypes: list[lbann.DataType]) -> lbann.DataType:
    """Return the widest datatype among a list of datatypes."""
    if not datatypes:
        raise ValueError('Empty list of datatypes provided')
    if (lbann.DataType.COMPLEX_FLOAT in datatypes
        or lbann.DataType.COMPLEX_DOUBLE in datatypes):
        raise ValueError('Not handling complex types in AMP')
    # If we support more complicated types, we may want to properly define
    # these relationships.
    widest = datatypes[0]
    for datatype in datatypes[1:]:
        if datatype == lbann.DataType.FLOAT and widest == lbann.DataType.FP16:
            widest = lbann.DataType.FLOAT
        elif datatype == lbann.DataType.DOUBLE:
            widest = lbann.DataType.DOUBLE  # Widest type.
        # No check for FP16, nothing gets promoted to it.
    return widest


def set_layer_datatypes(model: lbann.Model) -> None:
    """Set datatypes for layers in the model.

    If a layer that does not have a datatype set, it will be set based
    on the conversion lists and its parent layer types.

    """
    for layer in lbann.traverse_layer_graph(model.layers):
        if layer.datatype is not None:
            continue  # Skip when datatype is already set.
        layer_type = type(layer)
        if layer_type in FP16_LAYERS:
            layer.datatype = lbann.DataType.FP16
        elif layer_type in FP32_LAYERS:
            layer.datatype = lbann.DataType.FLOAT
        else:
            # Conservatively assume layers with no parents should
            # be in FP32 if there is not conversion or datatype
            # specified.
            if not layer.parents:
                layer.datatype = lbann.DataType.FLOAT
            else:
                # Set the layer's type as the widest type among its
                # parents.
                layer.datatype = get_widest_datatype(
                    [l.datatype for l in layer.parents])


def enable_amp(model: lbann.Model,
               args: argparse.ArgumentParser,
               init_scale: Optional[float] = None,
               growth_factor: Optional[float] = None,
               backoff_factor: Optional[float] = None,
               growth_interval: Optional[int] = None) -> None:
    """Enable automatic mixed precision for a model if requested."""
    if model.amp is not None:
        raise RuntimeError('Model already has AMP options set, not resetting')

    try:
        enable_amp = args.amp
    except AttributeError:
        raise ValueError('passed arguments have not been processed by '
                         '`add_amp_arguments`')

    if not enable_amp:
        return

    # Set up datatypes.
    add_weights(model)
    set_layer_datatypes(model)

    # Enable AMP in the model.
    model.amp = lbann.AmpOptions(
        enabled=True,
        init_scale=init_scale,
        growth_factor=growth_factor,
        backoff_factor=backoff_factor,
        growth_interval=growth_interval)
