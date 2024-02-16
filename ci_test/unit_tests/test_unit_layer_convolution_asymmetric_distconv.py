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
                      environment=lbann.contrib.args.get_distconv_environment(),
                      time_limit=10)
def test_simple(num_dims):
    try:
        import torch
        import torch.nn as nn
    except:
        pytest.skip('PyTorch is required to run this test.')

    torch.manual_seed(20240216)
    # Two samples of 4x16x16 or 4x16x16x16 tensors
    shape = [2, 4] + [16] * num_dims
    x = torch.randn(shape)
    if num_dims == 2:
        ConvClass = nn.Conv2d
        kerenel_size = (3, 1)
        padding = (1, 0)
        group_name = 'height_groups'
    else:
        ConvClass = nn.Conv3d
        kerenel_size = (5, 3, 1)
        padding = (2, 1, 0)
        group_name = 'depth_groups'

    conv = ConvClass(4, 8, kerenel_size, padding=padding, bias=False)
    with torch.no_grad():
        ref = conv(x)

    tester = test_util.ModelTester()
    x = tester.inputs(x.numpy())
    ref = tester.make_reference(ref.numpy())

    # Test layer
    kernel = conv.weight.detach().numpy()
    kernel_weights = lbann.Weights(
        initializer=lbann.ValueInitializer(values=np.nditer(kernel)),
        name=f'kernel_{num_dims}d'
    )
    ps = {group_name: tools.gpus_per_node(lbann)}
    y = lbann.Convolution(
        x,
        weights=(kernel_weights,),
        num_dims=num_dims,
        out_channels=8,
        kernel_size=kerenel_size,
        stride=1,
        padding=padding,
        dilation=1,
        has_bias=False,
        parallel_strategy=ps,
        name=f'conv_{num_dims}d'
    )
    y = lbann.Identity(y)
    tester.set_loss(lbann.MeanSquaredError(y, ref))
    tester.set_check_gradients_tensor(lbann.Square(y))
    return tester
