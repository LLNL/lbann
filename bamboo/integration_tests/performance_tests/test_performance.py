import pytest
from test_tools import *

def test_performance_lenet_mnist(dirname, exe):
  skeleton(dirname, exe, 'lenet_mnist', 'mnist', True)

def test_performance_alexnet(dirname, exe, weekly):
  if weekly:
    skeleton(dirname, exe, 'alexnet', 'imagenet', True)
  else:
    pytest.skip('Not doing weekly testing')
