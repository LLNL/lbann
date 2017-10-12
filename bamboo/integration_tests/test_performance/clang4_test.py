import os
import unittest

from test_tools import *

EXECUTABLE = 'NO_EXECUTABLE'

class Clang4Test(unittest.TestCase):

  def test_mnist_distributed_io(self):
    mnist_distributed_io_skeleton(self, EXECUTABLE)

  def test_alexnet(self):
    alexnet_skeleton(self, EXECUTABLE)

  def test_resnet50(self):
    resnet50_skeleton(self, EXECUTABLE)

if __name__ == '__main__':
  unittest.main()
