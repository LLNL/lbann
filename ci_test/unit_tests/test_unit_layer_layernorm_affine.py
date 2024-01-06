import lbann
import numpy as np
import test_util
import pytest


def numpy_layer_norm(x, scale=None, bias=None, eps=1e-5, dims=-1):
    mean = x.mean(axis=dims, keepdims=True)
    std = x.std(axis=dims, ddof=0, keepdims=True)
    result = (x - mean) / (std + eps)
    if scale is not None:
        result *= scale
    if bias is not None:
        result += bias
    return result


@test_util.lbann_test(check_gradients=True, train=True)
def test_layernorm_basic():
    np.random.seed(20230814)
    num_samples = 31
    sample_size = 31
    samples = np.random.normal(size=(num_samples,
                                     sample_size)).astype(np.float32)
    reference = numpy_layer_norm(samples)

    tester = test_util.ModelTester()
    x = tester.inputs(samples)
    ref = tester.make_reference(reference)

    y = lbann.LayerNorm(x)

    # Set test loss with a fixed tolerance (since all the values are close to
    # zero by design)
    tester.set_loss(lbann.MeanSquaredError(y, ref), tolerance=1e-8)
    tester.set_check_gradients_tensor(lbann.Square(y))
    return tester


@pytest.mark.parametrize('start_dim', (-1, 1, 0))
@test_util.lbann_test(check_gradients=True, train=True)
def test_layernorm_multidim(start_dim):
    np.random.seed(20230814)
    num_samples = 31
    sample_size = [31, 2, 5]
    sdim = (len(sample_size) + start_dim) if start_dim < 0 else start_dim
    dims = tuple(d+1 for d in range(sdim, len(sample_size)))
    samples = np.random.rand(num_samples, *sample_size).astype(np.float32)
    reference = numpy_layer_norm(samples, dims=dims)

    tester = test_util.ModelTester()
    x = tester.inputs(samples)
    ref = tester.make_reference(reference)

    y = lbann.LayerNorm(x, start_dim=start_dim)

    # Set test loss with a fixed tolerance (since all the values are close to
    # zero by design)
    tester.set_loss(lbann.MeanSquaredError(y, ref), tolerance=1e-7)
    tester.set_check_gradients_tensor(lbann.Square(y))
    return tester


@pytest.mark.parametrize('with_bias', (False, True))
@test_util.lbann_test(check_gradients=True, train=True)
def test_layernorm_scaled(with_bias):
    np.random.seed(20230814)
    num_samples = 31
    sample_size = 31
    samples = np.random.normal(size=(num_samples,
                                     sample_size)).astype(np.float32)
    scale = np.random.rand(sample_size).astype(np.float32)
    bias = np.random.rand(sample_size).astype(np.float32)
    reference = numpy_layer_norm(samples, scale, bias if with_bias else None)

    tester = test_util.ModelTester()
    x = tester.inputs(samples)
    reference = tester.make_reference(reference)

    if with_bias:
        y = lbann.LayerNorm(
            x,
            scale=True,
            bias=True,
            weights=[
                lbann.Weights(initializer=lbann.ValueInitializer(
                    values=np.nditer(scale))),
                lbann.Weights(initializer=lbann.ValueInitializer(
                    values=np.nditer(bias))),
            ])
    else:
        y = lbann.LayerNorm(
            x,
            scale=True,
            bias=False,
            weights=[
                lbann.Weights(initializer=lbann.ValueInitializer(
                    values=np.nditer(scale))),
            ])

    # Set test loss with a fixed tolerance (since all the values are close to
    # zero by design)
    tester.set_loss(lbann.MeanSquaredError(y, reference), tolerance=1e-8)
    tester.set_check_gradients_tensor(lbann.Square(y))
    return tester


@pytest.mark.skip  # No need to run since we have "scaled" which tests bias too
@test_util.lbann_test(check_gradients=True, train=True)
def test_layernorm_biasonly():
    np.random.seed(20230815)
    num_samples = 31
    sample_size = 31
    samples = np.random.normal(size=(num_samples,
                                     sample_size)).astype(np.float32)
    bias = np.random.rand(sample_size).astype(np.float32)
    reference = numpy_layer_norm(samples, bias=bias)

    tester = test_util.ModelTester()
    x = tester.inputs(samples)
    reference = tester.make_reference(reference)

    y = lbann.LayerNorm(x,
                        scale=False,
                        bias=True,
                        weights=[
                            lbann.Weights(initializer=lbann.ValueInitializer(
                                values=np.nditer(bias))),
                        ])

    # Set test loss with a fixed tolerance (since all the values are close to
    # zero by design)
    tester.set_loss(lbann.MeanSquaredError(y, reference), tolerance=1e-8)
    tester.set_check_gradients_tensor(lbann.Square(y))
    return tester


@test_util.lbann_test()
def test_layernorm_defaults():
    np.random.seed(20230814)
    num_samples = 31
    sample_size = 31
    samples = np.random.normal(size=(num_samples,
                                     sample_size)).astype(np.float32)
    reference = numpy_layer_norm(samples)

    tester = test_util.ModelTester()
    x = tester.inputs(samples)
    reference = tester.make_reference(reference)

    # Test that when no weights are given, scale initializes to 1 and bias to 0
    y = lbann.LayerNorm(x, scale=True, bias=True)

    # Set test loss with a fixed tolerance (since all the values are close to
    # zero by design)
    tester.set_loss(lbann.MeanSquaredError(y, reference), tolerance=1e-8)
    return tester
