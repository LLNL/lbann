import lbann
import numpy as np
import test_util
import pytest


@test_util.lbann_test(check_gradients=False)
def test_select_operator():
    # Prepare reference output
    np.random.seed(20230731)
    x1 = np.random.rand(20)
    x2 = np.random.rand(20)

    # Set every second element to 3
    predicate = np.random.rand(20)
    predicate[::2] = 3

    reference_numpy = np.copy(x2)
    reference_numpy[::2] = x1[::2]

    tester = test_util.ModelTester()

    predicate, x1, x2 = tester.inputs_like(predicate, x1, x2)
    reference = tester.make_reference(reference_numpy)

    # Test layer
    y = lbann.Select(predicate, x1, x2, value=3)

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    return tester


@pytest.mark.parametrize('true_or_false', (False, True))
@test_util.lbann_test(check_gradients=False)
def test_select_with_constant(true_or_false: bool):
    # Prepare reference output
    np.random.seed(20230731)
    x = np.random.rand(20)

    # Set every second element to 3
    predicate = np.random.rand(20)
    if true_or_false:
        predicate[::2] = 3
    else:
        predicate[1::2] = 3

    reference_numpy = np.copy(x)
    reference_numpy[::2] = 5

    tester = test_util.ModelTester()

    predicate, x = tester.inputs_like(predicate, x)
    reference = tester.make_reference(reference_numpy)

    # Test layer
    if true_or_false:
        y = lbann.Select(predicate, x, value=3, if_true=5)
    else:
        y = lbann.Select(predicate, x, value=3, if_false=5)

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    return tester


@test_util.lbann_test(check_gradients=False)
def test_select_all_constant():
    # Prepare reference output
    np.random.seed(20230731)

    # Set every second element to 3
    predicate = np.random.rand(20)
    predicate[::2] = 3

    reference_numpy = np.zeros([20])
    reference_numpy[::2] = 1

    tester = test_util.ModelTester()

    predicate = tester.inputs(predicate)
    reference = tester.make_reference(reference_numpy)

    # Test layer
    y = lbann.Select(predicate, value=3, if_false=0, if_true=1)

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    return tester
