from test_tools import *

def test_performance_mnist_distributed_io(exe, dirname):
  mnist_distributed_io_skeleton(exe, dirname)

def test_performance_alexnet(exe, dirname):
  alexnet_skeleton(exe, dirname)

def test_performance_resnet50(exe, dirname):
  resnet50_skeleton(exe, dirname)
