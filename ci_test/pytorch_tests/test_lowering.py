"""
Tests lowering functionality of the LBANN PyTorch frontend
"""
from typing import Any
import pytest

try:
    import torch
    if int(torch.__version__.split('.')[0]) < 2:
        raise ImportError('PyTorch < 2.0')
except (ModuleNotFoundError, ImportError):
    pytest.skip('PyTorch 2.0 is required for this test',
                allow_module_level=True)

from torch import nn
import lbann.torch
import numpy as np


def test_swish():
    """
    An activation function that LBANN does not have an equivalent of.
    """
    torch.manual_seed(20230620)

    class testmodule(nn.Module):

        def __init__(self):
            super().__init__()
            self.mod = nn.Hardswish()

        def forward(self, input):
            return self.mod(input)

    mod = testmodule()
    inp = torch.randn(1, 20)
    ref = mod(inp)

    # Expect a warning
    with pytest.warns(UserWarning,
                      match='Could not find replacement for module'):
        g = lbann.torch.compile(mod, input=inp)

    out = lbann.evaluate(
        g,
        inp.detach().numpy(),
        extra_callbacks=[lbann.CallbackPrintModelDescription()])

    assert torch.allclose(ref, torch.tensor(out))
