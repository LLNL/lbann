import numpy as np

import lbann

_permute_cache = {}
_cumsum_cache = {}


def Permute(x, dims, axes=None, name="", return_dims=False, avoid_cutensor=False):
    # (NOTE trb): I thought about adding additional checks for
    #   CPU-ness or the like, but I don't see that for the other cases
    #   here, so this implementation should match that behavior. That
    #   is, any layer output from this function will not have an
    #   explicit device allocation assigned to it. However, setting
    #   the device allocation of a TensorPermute layer to CPU will be
    #   fatal in the C++ runtime. So I've exposed a flag,
    #   'avoid_cutensor' to generate the gather-based permutation for
    #   this case. The user will still need to post-process the
    #   generated reshape and gather layers to set the device
    #   allocation as appropriate.
    if not avoid_cutensor and lbann.has_feature("CUTENSOR"):
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
