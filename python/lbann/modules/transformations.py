import numpy as np

import lbann

_permute_cache = {}
_cumsum_cache = {}


def Permute(x, dims, axes=None, name="", return_dims=False):
    global _permute_cache
    key = (dims, axes)
    size = np.prod(dims)
    if key not in _permute_cache:
        # Construct gather indices
        inds = np.arange(size).reshape(dims, order="C").transpose(axes)
        inds = lbann.Weights(
            initializer=lbann.ValueInitializer(
                values=np.nditer(inds, order="C"),
            ),
            optimizer=lbann.NoOptimizer(),
        )
        inds = lbann.WeightsLayer(dims=[size], weights=inds)
        _permute_cache[key] = inds

    # Apply transpose with gather
    inds = _permute_cache[key]
    if axes == None:
        new_dims = dims[::-1]
    else:
        new_dims = np.array(dims)[list(axes)]
    x = lbann.Reshape(x, dims=[size])
    y = lbann.Gather(x, inds)
    y = lbann.Reshape(y, dims=list(new_dims), name=name)

    if return_dims:
        return y, tuple(new_dims)
    return y


def Cumsum(x, dims, axis=0):
    global _cumsum_cache

    if len(dims) != 2:
        raise RuntimeError("dims > 2 not tested/supported for cumsum")
    if (axis < 0) or (axis > 1):
        raise RuntimeError("Unsupported cumsum axis: {}".format(axis))
    shape = (dims[axis], dims[axis])
    if shape not in _cumsum_cache:
        tril_ones = np.tril(np.full(shape, 1, dtype=int), k=0)
        tril_ones = lbann.Weights(
            initializer=lbann.ValueInitializer(
                values=np.nditer(tril_ones, order="C"),
            ),
            optimizer=lbann.NoOptimizer(),
        )
        tril_ones = lbann.WeightsLayer(dims=shape, weights=tril_ones)
        _cumsum_cache[shape] = tril_ones

    # Apply cumsum
    tril_ones = _cumsum_cache[shape]
    if axis == 0:
        x = lbann.MatMul(tril_ones, x)
        return x
    if axis == 1:
        x = lbann.MatMul(x, tril_ones, transpose_b=True)
        return x

def PeriodicPadding2D(x, height, width, padding=1):
    """ For 2D images of the shape (B, *, height, width)
        Args:
            x (lbann.Layer): input tensor to padded of the shape (*, height, width)
            height (int): 2nd dimension of the 4D tensor
            width (int): 3rd dimension of the 4D tensor
            padding (int): The amount to pad (default: 1)
        returns:
            (lbann.Layer): Returns periodically padded layer of
                           shape (*, height + padding, width + padding)
    """
    horizontal_slices = lbann.Slice(x,
                                    slice_points=str_list([0, padding, height-padding, height]),
                                    axis=1)
    top = lbann.Identity(horizontal_slices)
    _ = lbann.Identity(horizontal_slices)
    bottom = lbann.Identity(horizontal_slices)

    x = lbann.Concatenation([top, x, bottom], axis=1)

    vertical_slices = lbann.Slice(x,
                                  slice_points=str_list([0, padding, width-padding, width]),
                                  axis=2)
    left = lbann.Identity(vertical_slices)
    _ = lbann.Identity(vertical_slices)
    right = lbann.Identity(vertical_slices)

    x = lbann.Concatenation([left, x, right], axis=2)
    return x

def PeriodicPadding3D(x, height, width, depth, padding=1):
    """ For 3D volumes of the shape (B, *, depth, height, width)
        Args:
            x (lbann.Layer): input tensor to padded of the shape (*, depth, height, width)
            depth (int): 1st dimension of the 4D tensor
            height (int): 2nd dimension of the 4D tensor
            width (int): 3rd dimension of the 4D tensor
            padding (int): The amount to pad (default: 1)
        returns:
            (lbann.Layer): Returns periodically padded layer of
                           shape (*, depth + padding, height + padding, width + padding)
    """
    depth_slices = lbann.Slice(x,
                               slice_points=str_list([0, padding, depth-padding, depth]),
                               axis=1)
    d1 = lbann.Identity(depth_slices)
    _ = lbann.Identity(depth_slices)
    d2 = lbann.Identity(depth_slices)

    x = lbann.Concatenation([d1, x, d2], axis=1)

    height_slices = lbann.Slice(x,
                                slice_points=str_list([0, padding, height-padding, height]),
                                axis=2)
    h1 = lbann.Identity(height_slices)
    _ = lbann.Identity(height_slices)
    h2 = lbann.Identity(height_slices)

    width_slices = lbann.Slice(x,
                               slice_points=str_list([0, padding, width-padding, width]),
                               axis=3)
    w1 = lbann.Identity(width_slices)
    _ = lbann.Identity(width_slices)
    w2 = lbann.Identity(width_slices)

    x = lbann.Concatenation([w1, x, w2], axis=3)
    return x