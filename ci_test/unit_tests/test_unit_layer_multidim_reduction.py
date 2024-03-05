import lbann
import numpy as np
import test_util
import pytest


@test_util.lbann_test(check_gradients=True)
def test_multidim_reduction():
    if not lbann.has_feature('CUTENSOR'):
        pytest.skip('Test requires LBANN to be built with cuTENSOR')

    # Prepare reference output
    np.random.seed(20240228)
    shape = [25, 3, 4, 5, 6]
    x = np.random.rand(*shape).astype(np.float32)
    ref = x.sum(axis=(2, 4))

    tester = test_util.ModelTester()

    x = tester.inputs(x)
    reference = tester.make_reference(ref)

    # Test layer
    # Note that the axes here are different (as the mini-batch dimension is
    # ignored).
    y = lbann.MultiDimReduction(x, axes=(1, 3))

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    tester.set_check_gradients_tensor(lbann.Square(y))
    return tester


@test_util.lbann_test(check_gradients=False)
def test_multidim_reduction_max():
    if not lbann.has_feature('CUTENSOR'):
        pytest.skip('Test requires LBANN to be built with cuTENSOR')

    # Prepare reference output
    np.random.seed(20240228)
    shape = [25, 3, 4, 5, 6]
    x = np.random.rand(*shape).astype(np.float32)
    ref = x.max(axis=(3, 1))

    tester = test_util.ModelTester()

    x = tester.inputs(x)
    reference = tester.make_reference(ref)

    # Test layer
    y = lbann.MultiDimReduction(x, axes=(2, 0), mode='max')

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    return tester
