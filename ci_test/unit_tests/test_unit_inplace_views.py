import lbann
import numpy as np
import test_util
import pytest


@test_util.lbann_test(check_gradients=False)
def test_inplace_view():
    # Prepare reference output
    np.random.seed(20230606)
    # Note: The leaky ReLU is not differentiable at 0, so we make sure values
    # are away from 0.
    num_samples = 23
    sample_size = 48
    samples = np.random.choice([-1.0, 1.0], size=(num_samples, sample_size))
    samples += np.random.uniform(-0.5, 0.5, size=samples.shape)
    ref = np.where(samples > 0, samples, 2 * samples)

    tester = test_util.ModelTester()

    x_lbann = tester.inputs(samples)
    reference = tester.make_reference(ref)

    # LBANN implementation
    x = lbann.Reshape(x_lbann, dims=[4, 2, 6])
    y = lbann.LeakyRelu(x, negative_slope=2)
    y = lbann.Reshape(y, dims=[48])


    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    return tester
