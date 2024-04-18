import numpy as np

import lbann

_permute_cache = {}
_cumsum_cache = {}


def Permute(x, dims, axes=None, name="", return_dims=False, avoid_gpu_permute=False):
    # (NOTE trb): I thought about adding additional checks for
    #   CPU-ness or the like, but I don't see that for the other cases
    #   here, so this implementation should match that behavior. That
    #   is, any layer output from this function will not have an
    #   explicit device allocation assigned to it. However, setting
    #   the device allocation of a TensorPermute layer to CPU will be
    #   fatal in the C++ runtime. So I've exposed a flag,
    #   'avoid_gpu_permute' to generate the gather-based permutation for
    #   this case. The user will still need to post-process the
    #   generated reshape and gather layers to set the device
    #   allocation as appropriate.
    if not avoid_gpu_permute and lbann.has_feature("TENSOR_PERMUTE"):
        ndims = len(dims)
        if axes is None: # Apparently this means "reverse things".
            axes=[x for x in range(0, len(dims))]
            axes.reverse()
        axes = [ ndims + x if x < 0 else x for x in axes ]
        y = lbann.TensorPermute(x, axes=list(axes))
        if return_dims:
            new_dims = np.array(dims)[list(axes)]
            return y, tuple(new_dims)
        return y
    else:
        global _permute_cache
        key = (dims, axes)
        size = np.prod(dims)
        if key not in _permute_cache:
            # Construct gather indices
            inds = np.arange(size).reshape(dims, order="C").transpose(axes)
            inds = lbann.Weights(
                initializer=lbann.ValueInitializer(
                    values=[float(x) for x in np.nditer(inds, order="C")],
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
                values=[float(x) for x in np.nditer(tril_ones, order="C")],
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
    """ For 2D images of the shape (channels, height, width)
        Args:
            x (lbann.Layer): input tensor to padded of the shape (channels, height, width)
            height (int): 1st dimension of the 3D tensor
            width (int): 2nd dimension of the 3D tensor
            padding (int): The amount to pad on each side (default: 1)
        returns:
            (lbann.Layer): Returns periodically padded layer of
                           shape (channels, height + 2 * padding, width + 2 * padding)
    """
    horizontal_slices = lbann.Slice(x,
                                    slice_points=[0, padding, height - padding, height],
                                    axis=1)
    top = lbann.Identity(horizontal_slices)
    _ = lbann.Identity(horizontal_slices)
    bottom = lbann.Identity(horizontal_slices)

    x = lbann.Concatenation([bottom, x, top], axis=1)

    vertical_slices = lbann.Slice(x,
                                  slice_points=[0, padding, width - padding, width],
                                  axis=2)
    left = lbann.Identity(vertical_slices)
    _ = lbann.Identity(vertical_slices)
    right = lbann.Identity(vertical_slices)

    x = lbann.Concatenation([right, x, left], axis=2)
    return x


def PeriodicPadding3D(x, depth, height, width, padding=1):
    """ For 3D volumes of the shape (channels, depth, height, width)
        Args:
            x (lbann.Layer): input tensor to be padded of the shape (channels, depth, height, width)
            depth (int): 1st dimension of the 4D tensor
            height (int): 2nd dimension of the 4D tensor
            width (int): 3rd dimension of the 4D tensor
            padding (int): The amount to pad (default: 1)
        returns:
            (lbann.Layer): Returns periodically padded layer of
                           shape (channels, depth + 2 * padding, height + 2 * padding, width + 2 * padding)
    """
    #  To do: Hack to get around slice and concatenation limitation. Remove when
    #         support for arbitrary dimensional slice + concatenation is added
    x = lbann.Reshape(x, dims=[-1, depth, height * width])  # Shape (C, D, H * W)
    depth_slices = lbann.Slice(x,
                               slice_points=[0, padding, depth - padding, depth],
                               axis=1)
    d1 = lbann.Identity(depth_slices)
    _ = lbann.Identity(depth_slices)
    d2 = lbann.Identity(depth_slices)

    x = lbann.Concatenation([d2, x, d1], axis=1)

    #  To do: Hack to get around slice and concatenation limitation. Remove when
    #         support for arbitrary dimensional slice + concatenation is added
    x = lbann.Reshape(x, dims=[-1, height, width])  # Shape (C * D, H ,  W)
    height_slices = lbann.Slice(x,
                                slice_points=[0, padding, height - padding, height],
                                axis=1)
    h1 = lbann.Identity(height_slices)
    _ = lbann.Identity(height_slices)
    h2 = lbann.Identity(height_slices)

    x = lbann.Concatenation([h2, x, h1], axis=1)

    width_slices = lbann.Slice(x,
                               slice_points=[0, padding, width - padding, width],
                               axis=2)
    w1 = lbann.Identity(width_slices)
    _ = lbann.Identity(width_slices)
    w2 = lbann.Identity(width_slices)

    x = lbann.Concatenation([w2, x, w1], axis=2)

    #  To do: Hack to get around slice and concatenation limitation. Remove when
    #         support for arbitrary dimensional slice + concatenation is added
    x = lbann.Reshape(x, dims=[-1,
                               depth + 2 * padding,
                               height + 2 * padding,
                               width + 2 * padding])
    return x
