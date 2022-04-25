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
