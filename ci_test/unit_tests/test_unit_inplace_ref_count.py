import lbann
import numpy as np
import test_util
import pytest


@test_util.lbann_test(check_gradients=True)
def test_inplace_ref_count():
    # Prepare reference output
    np.random.seed(20240430)
    # Note: ReLU is not differentiable at 0, so we make sure values
    # are away from 0.
    num_samples = 24
    sample_size = 48
    samples = np.random.choice([-1.0, 1.0], size=(num_samples, sample_size))
    samples += np.random.uniform(-0.5, 0.5, size=samples.shape)
    ref = np.maximum(0, samples)

    tester = test_util.ModelTester()

    x_lbann = tester.inputs(samples)
    reference = tester.make_reference(ref)

    # LBANN implementation:
    # The first relu will run in-place and then decref its inputs since it
    # doesn't need them for backprop. The second relu will then do the same.
    # If the in-place layer's output (same buffer as input) is properly
    # reference counted, then it will not be freed before it is needed for the
    # in-place layer's backprop.
    x = lbann.Relu(x_lbann)
    x = lbann.Relu(x)

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(x, reference))
    tester.set_check_gradients_tensor(x)
    return tester
