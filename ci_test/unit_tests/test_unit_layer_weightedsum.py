import lbann
import numpy as np
import test_util
import pytest


@test_util.lbann_test(check_gradients=False)
def test_weightedsum_twoinputs():
    # Prepare reference output
    np.random.seed(20230516)
    x1 = np.random.rand(20, 20)
    x2 = np.random.rand(20, 20)
    reference_numpy = 0.25 * x1 + 0.5 * x2

    tester = test_util.ModelTester()

    x1, x2 = tester.inputs_like(x1, x2)
    reference = tester.make_reference(reference_numpy)

    # Test layer
    y = lbann.WeightedSum(x1, x2, scaling_factors=[0.25, 0.5])

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    return tester


@pytest.mark.parametrize('inputs', [3, 5])
@test_util.lbann_test(check_gradients=False)
def test_weightedsum_n_inputs(inputs):
    # Prepare reference output
    np.random.seed(20230516)
    x = [np.random.rand(3, 20) for _ in range(inputs)]
    factors = [np.random.rand() for _ in range(inputs)]
    reference_numpy = sum(f * xi for f, xi in zip(factors, x))

    tester = test_util.ModelTester()

    x = tester.inputs_like(*x)
    reference = tester.make_reference(reference_numpy)

    # Test layer
    y = lbann.WeightedSum(*x, scaling_factors=factors)

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    tester.tolerance *= inputs  # Make it more tolerant towards more inputs
    return tester


@pytest.mark.parametrize('dims', [1, 3])
@test_util.lbann_test(check_gradients=False)
def test_weightedsum_oneinput(dims):
    # Prepare reference output
    np.random.seed(20230516)
    shape = [3] + [2] * dims
    x = np.random.rand(*shape).astype(np.float32)
    a = 0.4
    ref = a * x

    tester = test_util.ModelTester()

    x = tester.inputs(x)
    reference = tester.make_reference(ref)

    # Test layer
    y = lbann.WeightedSum(x, scaling_factors=[a])

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    return tester
