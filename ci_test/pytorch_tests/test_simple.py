"""
Tests simple compiled functions and nn.Modules with the LBANN PyTorch frontend
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


def test_simple_function():

    @lbann.torch.lazy_compile
    def fn(x):
        a = torch.cos(x)
        b = torch.sin(x) + 2
        return a + b

    # Obtain graph
    a = torch.randn(1, 20)
    graph = fn(a)

    # Test correctness
    reference = torch.cos(a) + torch.sin(a) + 2
    output = lbann.evaluate(graph, a.numpy())
    assert torch.allclose(reference, torch.tensor(output))


def test_simple_module():

    class mymod(nn.Module):

        def __init__(self, func=lambda a, b: a + b):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x, x)

    # Challenging setup where a lambda function does not reside in the same
    # line and modified at runtime.
    a = mymod()
    a.func = lambda x,y:\
        x*y+5

    # Create graph
    g = lbann.torch.compile(a, x=torch.randn(20, 20))

    # Compile and run graph with numpy
    inp = np.random.rand(20, 20).astype(np.float32)
    ref = inp * inp + 5
    out = lbann.evaluate(
        g, inp, extra_callbacks=[lbann.CallbackPrintModelDescription()])

    # Test correctness
    assert np.allclose(out, ref)


def test_module_with_parameters():

    class parameterized(nn.Module):

        def __init__(self, func=lambda a, b: a + b):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    mod = parameterized()

    # Run PyTorch version
    with torch.no_grad():
        inp = torch.randn(2, 1, 3, 3)
        ref = mod(inp)

    # Compile and run LBANN graph
    g = lbann.torch.compile(mod, x=inp)
    out = lbann.evaluate(
        g,
        inp.numpy(),
        extra_callbacks=[lbann.CallbackPrintModelDescription()])

    # Test correctness
    assert torch.allclose(ref, torch.tensor(out))


def test_module_with_parameters_custom():

    class parameterized(nn.Module):

        def __init__(self, func=lambda a, b: a + b):
            super().__init__()
            self.p = nn.Parameter(torch.randn(1, 20))

        def forward(self, x):
            return x * self.p

    mod = parameterized()

    # Create graph
    g = lbann.torch.compile(mod, x=torch.randn(20))

    # Compile and run graph with numpy
    inp = np.random.rand(1, 20).astype(np.float32)
    ref = inp * mod.p.detach().numpy()
    out = lbann.evaluate(
        g, inp, extra_callbacks=[lbann.CallbackPrintModelDescription()])

    # Test correctness
    assert np.allclose(out, ref)


if __name__ == '__main__':
    test_simple_function()
    test_simple_module()
    test_module_with_parameters()
    test_module_with_parameters_custom()
