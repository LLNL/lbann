"""
Testing correctenss of the LBANN PyTorch frontend with LeNet-5 for evaluation
and training.
"""
import pytest

try:
    import torch
    if int(torch.__version__.split('.')[0]) < 2:
        raise ImportError('PyTorch < 2.0')
except (ModuleNotFoundError, ImportError):
    pytest.skip('PyTorch 2.0 is required for this test',
                allow_module_level=True)

from torch import nn
import torch.nn.functional as F

import lbann
import lbann.torch
import lbann.contrib.launcher

# Import MNIST dataset from applications
import sys
import os

current_file = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file)
path_to_datasets = os.path.join(current_dir, '..', '..', 'applications',
                                'vision')
sys.path.insert(0, path_to_datasets)
import data.mnist


def convsize(insize, pad, stride, dilation, kernel):
    return ((insize + 2 * pad - dilation * (kernel - 1) - 1) // stride) + 1


class LeNet5(nn.Module):
    """
    LeNet-5, as presented in Y. LeCun et al., "Gradient-Based Learning Applied
    to Document Recognition", Proc. IEEE 1998.
    """

    def __init__(self, size=28, classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        size = convsize(size, 2, 1, 1, 5)
        self.pool1 = nn.AvgPool2d(2, 2)
        size //= 2
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        size = convsize(size, 0, 1, 1, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        size //= 2
        self.dense1 = nn.Linear(size * size * 16, 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, classes)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1, -1)
        x = F.tanh(self.dense1(x))
        x = F.tanh(self.dense2(x))
        x = self.dense3(x)
        return F.softmax(x, dim=-1)


def test_lenet_eval():
    torch.manual_seed(20230621)
    B = 8
    mod = LeNet5()

    # Evaluate on random inputs
    inputs = torch.rand(B, 1, 28, 28)
    reference = mod(inputs)

    g = lbann.torch.compile(mod, x=torch.rand(B, 1, 28, 28))
    outputs = lbann.evaluate(g, inputs)

    assert torch.allclose(reference, torch.tensor(outputs))


def test_lenet_train():
    torch.manual_seed(20230621)
    B = 8
    mod = LeNet5()

    # Get initial graph
    graph = lbann.torch.compile(mod, x=torch.rand(B, 1, 28, 28))
    images = graph[0]
    probs = graph[-1]

    # Create loss function and accuracy
    labels = lbann.Input(data_field='labels')
    loss = lbann.CrossEntropy(probs, labels)
    acc = lbann.CategoricalAccuracy(probs, labels)

    # Setup model
    num_epochs = 5
    model = lbann.Model(num_epochs,
                        layers=lbann.traverse_layer_graph([images, labels]),
                        objective_function=loss,
                        metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                        callbacks=[lbann.CallbackPrint()])

    # Setup optimizer
    opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

    # Setup data reader
    data_reader = data.mnist.make_data_reader()

    # Setup trainer
    trainer = lbann.Trainer(mini_batch_size=64)

    # Run experiment
    lbann.contrib.launcher.run(trainer,
                               model,
                               data_reader,
                               opt,
                               job_name='lenet_pytorch_test')
