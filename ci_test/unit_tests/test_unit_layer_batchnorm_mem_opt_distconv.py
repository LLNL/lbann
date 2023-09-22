import lbann
import numpy as np
import test_util
import pytest
import os
import sys
import lbann.contrib.launcher
import lbann.contrib.args

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

@pytest.mark.parametrize('num_dims', [2, 3])
@test_util.lbann_test(check_gradients=True,
                      train=True,
                      environment=lbann.contrib.args.get_distconv_environment(),
                      time_limit=5)
def test_simple(num_dims):
    np.random.seed(20230621)
    # Two samples of 4x16x16 or 4x16x16x16 tensors
    shape = [2, 4] + [16] * num_dims
    x = np.random.standard_normal(shape).astype(np.float32)
    dims = (0,2,3) if num_dims == 2 else (0,2,3,4)
    ref = (x - x.mean(axis=dims, keepdims=True)) / np.sqrt(x.var(axis=dims, keepdims=True, ddof=1) + 1e-5)

    tester = test_util.ModelTester()
    x = tester.inputs(x)
    ref = tester.make_reference(ref)

    # Test layer
    ps = {'height_groups': tools.gpus_per_node(lbann)}
    y = lbann.BatchNormalization(
        x,
        parallel_strategy=ps,
        name=f'batchnorm_{num_dims}d'
    )
    y = lbann.Identity(y)
    tester.set_loss(lbann.MeanSquaredError(y, ref), tolerance=5e-7)
    tester.set_check_gradients_tensor(lbann.Square(y))
    return tester
