"""
Tests a reference implementation of CosmoFlow with the LBANN PyTorch frontend.
"""
import pytest
import math

try:
    import torch
    if int(torch.__version__.split('.')[0]) < 2:
        raise ImportError('PyTorch < 2.0')
except (ModuleNotFoundError, ImportError):
    pytest.skip('PyTorch 2.0 is required for this test',
                allow_module_level=True)

try:
    import timm
except (ModuleNotFoundError, ImportError):
    pytest.skip('Torch Image Models (timm) is required for this test',
                allow_module_level=True)

import lbann
import lbann.torch
from torch import nn


def test_batchnorm():
    torch.manual_seed(20230622)

    class module(nn.Module):

        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3, eps=1e-5, momentum=0.1)

        def forward(self, x):
            return self.bn(x)

    mod = module()
    n = 8
    x = torch.randn(n, 3, 2, 1)
    reference = mod(x)

    g = lbann.torch.compile(mod, x=x)
    # training=True is required to evaluate batch normalization
    outputs = lbann.evaluate(g, x.detach().numpy(), training=True)

    # Loose tolerance to account for differences in implementations
    assert torch.allclose(reference,
                          torch.tensor(outputs),
                          atol=1e-1,
                          rtol=1e-1)


def test_resnet_34():
    torch.manual_seed(20230622)

    # Test both architecture and loading pretrained weights
    mod = timm.create_model('resnet34', pretrained=True)

    x = torch.randn(32, 3, 32, 32)
    reference = mod(x)

    g = lbann.torch.compile(mod, x=x)
    outputs = lbann.evaluate(g, x.detach().numpy(), training=True)

    # Using loose tolerance to account for full model differences in operator
    # implementations and determinism
    assert torch.allclose(reference, torch.tensor(outputs), atol=1e1, rtol=1.0)
