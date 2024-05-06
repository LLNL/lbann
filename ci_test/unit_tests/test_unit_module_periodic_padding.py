import lbann
import lbann.modules as lm
import numpy as np
import test_util
import pytest

try:
    from torch import Tensor
    import torch.nn.functional as F
except:
    pytest.skip("PyTorch is required to run this test.", allow_module_level=True)


@test_util.lbann_test(check_gradients=False)
def test_periodic_padding_2D():
    # Prepare reference output
    np.random.seed(20240228)
    shape = [1, 4, 16, 20]
    _, _, height, width = shape
    x = np.random.rand(*shape).astype(np.float32)
    ref = F.pad(Tensor(x), (1,1,1,1), mode="circular").numpy()

    tester = test_util.ModelTester()

    x = tester.inputs(x)[0]
    reference = tester.make_reference(ref)
    # Test layer
    y = lm.PeriodicPadding2D(x, height=height, width=width, padding=1)

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    return tester

@test_util.lbann_test(check_gradients=False)
def test_periodic_padding_3D():
    # Prepare reference output
    np.random.seed(20240228)
    shape = [1, 4, 8, 16, 20]
    _, _, depth, height, width = shape
    x = np.random.rand(*shape).astype(np.float32)
    ref = F.pad(Tensor(x), (1,1,1,1,1,1), mode="circular").numpy()

    tester = test_util.ModelTester()

    x = tester.inputs(x)[0]
    reference = tester.make_reference(ref)
    # Test layer
    y = lm.PeriodicPadding3D(x, depth=depth, height=height, width=width, padding=1)

    # Set test loss
    tester.set_loss(lbann.MeanSquaredError(y, reference))
    return tester
