import os
import unittest

from test_tools import *

EXECUTABLE = 'NO_EXECUTABLE'
# EXECUTABLE = '../LF-ND-SS/opt/spack/linux-rhel7-x86_64/gcc-4.9.3/lbann-develop-6krf36jbmunljhcrzuz4qo7kc3fqrept/lib64/liblbann.so'

class Gcc4Test(unittest.TestCase):

  def test_mnist_distributed_io(self):
    mnist_distributed_io_skeleton(self, EXECUTABLE)

  def test_alexnet(self):
    alexnet_skeleton(self, EXECUTABLE)

  def test_resnet50(self):
    resnet50_skeleton(self, EXECUTABLE)

if __name__ == '__main__':
  unittest.main()
