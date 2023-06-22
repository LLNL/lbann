"""
Tests a reference implementation of CosmoFlow with the LBANN PyTorch frontend.
"""
import pytest

try:
    import torch
    if int(torch.__version__.split('.')[0]) < 2:
        raise ImportError('PyTorch < 2.0')
except (ModuleNotFoundError, ImportError):
    pytest.skip('PyTorch 2.0 is required for this test',
                allow_module_level=True)

import lbann
import lbann.torch

# Import the reference CosmoFlow module
import os
import sys

current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
path_to_cosmoflow = os.path.join(current_dir, '..', '..', 'applications',
                                 'physics', 'cosmology', 'cosmoflow', 'ref')
sys.path.insert(0, path_to_cosmoflow)
from model import CosmoFlow


def test_cosmoflow():
    mod = CosmoFlow().cuda()
    x = torch.randn(1, 4, 128, 128, 128).cuda()
    reference = mod(x)

    g = lbann.torch.compile(mod, x=x)
    outputs = lbann.evaluate(g, x.detach().cpu().numpy())

    # Using loose tolerance values due to running a full network on 128^3 inputs
    assert torch.allclose(reference.cpu(),
                          torch.tensor(outputs),
                          rtol=1e-2,
                          atol=1e-2)
