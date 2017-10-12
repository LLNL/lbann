import os
import re
import unittest

from test_tools import *

def executable(hostname_file_name='hostname_output.txt', top_level_file_name='top_level_output.txt'):
  os.system('hostname > %s' % hostname_file_name)
  hostname_file = open(hostname_file_name, 'r')
  hostname = hostname_file.readline().rstrip()
  hostname_file.close()
  os.system('rm %s' % hostname_file_name)

  os.system('echo $(git rev-parse --show-toplevel) > %s' % top_level_file_name)
  top_level_file = open(top_level_file_name, 'r')
  top_level = top_level_file.readline().rstrip()
  top_level_file.close()
  os.system('rm %s' % top_level_file_name)

  cluster = re.sub('[0-9]+', '.llnl.gov', hostname)
  return '%s/build/%s/model_zoo/lbann' % (top_level, cluster)

EXECUTABLE = executable()

class ScriptTest(unittest.TestCase):

  def test_mnist_distributed_io(self):
    mnist_distributed_io_skeleton(self, EXECUTABLE)

  def test_alexnet(self):
    alexnet_skeleton(self, EXECUTABLE)

  def test_resnet50(self):
    resnet50_skeleton(self, EXECUTABLE)

if __name__ == '__main__':
  unittest.main()
