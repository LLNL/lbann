import lbann
import numpy as np
import test_util
import pytest
import os
import sys
import lbann.contrib.launcher

# Bamboo utilities
current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'common_python'))
import tools

@pytest.mark.parametrize('num_dims', [2, 3])
@test_util.lbann_test(check_gradients=True, environment=tools.get_distconv_environment())
def test_simple(num_dims):
    np.random.seed(20230607)
    # Two samples of 4x16x16 or 4x16x16x16 tensors
    shape = [2, 4] + [16] * num_dims
    x = np.random.standard_normal(shape).astype(np.float32)
    ref = x * ((x > 0) + 0.2 * (x < 0))

    tester = test_util.ModelTester()
    x = tester.inputs(x)
    ref = tester.make_reference(ref)

    # Test layer
    kernel_shape = [4, 4] + [3] * num_dims
    kernel = np.zeros(kernel_shape)
    for i in range(kernel_shape[0]):
        if num_dims == 2:
            kernel[i,i,1,1] = 1
        else:
            kernel[i,i,1,1,1] = 1
    kernel_weights = lbann.Weights(
        initializer=lbann.ValueInitializer(values=np.nditer(kernel)),
        name=f'kernel_{num_dims}d'
    )
    ps = {'height_groups': tools.gpus_per_node(lbann)}
    y = lbann.Convolution(
        x,
        weights=(kernel_weights,),
        num_dims=num_dims,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        has_bias=False,
        parallel_strategy=ps,
        name=f'conv_{num_dims}d'
    )
    z = lbann.LeakyRelu(y, negative_slope=0.2, parallel_strategy=ps,
                        name=f'leaky_relu_{num_dims}d')
    tester.set_loss(lbann.MeanSquaredError(z, ref))
    return tester
