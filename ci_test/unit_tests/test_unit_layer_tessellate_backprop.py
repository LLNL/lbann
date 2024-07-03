import lbann
import numpy as np
import test_util


@test_util.lbann_test(check_gradients=True)
def test_tessellate_pad():
    # Prepare reference output
    np.random.seed(20240627)
    x = np.random.rand(37, 20, 1)
    reference_numpy = np.tile(x, (1, 1, 5))

    tester = test_util.ModelTester()

    x = tester.inputs_like(x)
    reference = tester.make_reference(reference_numpy)

    # Test layer
    y = lbann.Tessellate(x, dims=[20, 5])

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    tester.set_check_gradients_tensor(lbann.Square(y))

    return tester


@test_util.lbann_test(check_gradients=True)
def test_tessellate_scalar():
    # Prepare reference output
    np.random.seed(20240627)
    x = np.random.rand(37, 1, 1, 1)
    reference_numpy = np.tile(x, (1, 5, 6, 7))

    tester = test_util.ModelTester()

    x = tester.inputs_like(x)
    reference = tester.make_reference(reference_numpy)

    # Test layer
    y = lbann.Tessellate(x, dims=[5, 6, 7])

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    tester.set_check_gradients_tensor(lbann.Square(y))

    return tester


@test_util.lbann_test(check_gradients=True)
def test_tessellate_repro1():
    # Prepare reference output
    np.random.seed(20240627)
    x = np.random.rand(16, 3, 1, 1)
    reference_numpy = np.tile(x, (1, 16, 3, 3))

    tester = test_util.ModelTester()

    x = tester.inputs_like(x)
    reference = tester.make_reference(reference_numpy)

    # Test layer
    y = lbann.Tessellate(x, dims=[3 * 16, 3, 3])

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    tester.set_check_gradients_tensor(lbann.Square(y))

    return tester


@test_util.lbann_test(check_gradients=True)
def test_tessellate_repro2():
    # Prepare reference output
    np.random.seed(20240627)
    x = np.random.rand(37, 16, 1, 1)
    reference_numpy = np.tile(x, (1, 1, 64, 64))

    tester = test_util.ModelTester()

    x = tester.inputs_like(x)
    reference = tester.make_reference(reference_numpy)

    # Test layer
    y = lbann.Tessellate(x, dims=[16, 64, 64])

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    tester.set_check_gradients_tensor(lbann.Square(y))

    return tester
