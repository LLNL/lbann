################################################################################
# Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. <lbann-dev@llnl.gov>
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LLNL/LBANN.
#
# Licensed under the Apache License, Version 2.0 (the "Licensee"); you
# may not use this file except in compliance with the License.  You may
# obtain a copy of the License at:
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the license.
#
################################################################################
import lbann
from dataclasses import dataclass, field
import functools
import inspect
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import os
import re
import tools
import lbann.contrib.single_tensor_data_reader as single_tensor_data_reader


def lbann_test(check_gradients=False, train=False, **decorator_kwargs):
    """
    A decorator that wraps an LBANN-enabled model unit test.
    Use it before a function named ``test_*`` to run it automatically in pytest.
    The unit test in the wrapped function must return a ``test_util.ModelTester``
    object, which contains all the necessary information to test the model (e.g.,
    model, input/reference tensors).

    The decorator wraps the test with the appropriate setup phase, data reading,
    callbacks, and metrics so that the test functions properly.
    """

    def internal_tester(f):

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            # Call model constructor
            tester = f(*args, **kwargs)

            # Check return value
            if not isinstance(tester, ModelTester):
                raise ValueError('LBANN test must return a ModelTester object')
            if tester.loss is None:
                raise ValueError(
                    'LBANN test did not define a loss function, '
                    'use ``ModelTester.set_loss`` or ``set_loss_function``.')
            if tester.input_tensor is None:
                raise ValueError('LBANN test did not define an input, call '
                                 '``ModelTester.inputs`` or ``inputs_like``.')
            if (tester.reference_tensor is not None
                    and tester.reference_tensor.shape[0] !=
                    tester.input_tensor.shape[0]):
                raise ValueError(
                    'Input and reference tensors in LBANN test '
                    'must match in the first (minibatch) dimension')
            full_graph = lbann.traverse_layer_graph(tester.loss)
            callbacks = []
            callbacks.append(
                lbann.CallbackCheckMetric(metric='test',
                                          lower_bound=0,
                                          upper_bound=tester.tolerance,
                                          error_on_failure=True,
                                          execution_modes='train' if train else 'test'))

            check_grad_obj_func = None
            if check_gradients:
                if tester.check_gradients_tensor is None:
                    raise ValueError(
                        'LBANN test did not set a tensor for checking gradients, '
                        'use ``ModelTester.set_check_gradients_tensor``.')
                check_grad_obj_func = tester.check_gradients_tensor
                callbacks.append(
                    lbann.CallbackCheckGradients(error_on_failure=True))
            callbacks.extend(tester.extra_callbacks)

            metrics = [lbann.Metric(tester.loss, name='test')]
            metrics.extend(tester.extra_metrics)
            model = lbann.Model(epochs=1 if train else 0,
                                layers=full_graph,
                                objective_function=check_grad_obj_func if check_gradients else tester.loss,
                                metrics=metrics,
                                callbacks=callbacks)

            # Get file
            file = inspect.getfile(f)

            def setup_func(lbann, weekly):
                # Get minibatch size from tensor
                mini_batch_size = tester.input_tensor.shape[0]

                # Save combined input/reference data to file
                work_dir = _get_work_dir(file)
                os.makedirs(work_dir, exist_ok=True)
                if tester.reference_tensor is not None:
                    flat_inp = tester.input_tensor.reshape(mini_batch_size, -1)
                    flat_ref = tester.reference_tensor.reshape(
                        mini_batch_size, -1)
                    np.save(os.path.join(work_dir, 'data.npy'),
                            np.concatenate((flat_inp, flat_ref), axis=1))
                else:
                    np.save(os.path.join(work_dir, 'data.npy'),
                            tester.input_tensor.reshape(mini_batch_size, -1))

                # Setup data reader
                data_reader = lbann.reader_pb2.DataReader()
                data_reader.reader.extend([
                    tools.create_python_data_reader(
                        lbann, single_tensor_data_reader.__file__,
                        'get_sample', 'num_samples', 'sample_dims', 'train')
                ])
                if not train:
                    data_reader.reader.extend([
                        tools.create_python_data_reader(
                            lbann, single_tensor_data_reader.__file__,
                            'get_sample', 'num_samples', 'sample_dims', 'test')
                    ])

                trainer = lbann.Trainer(mini_batch_size)
                optimizer = lbann.SGD(learn_rate=0)
                return trainer, model, data_reader, optimizer, None  # Don't request any specific number of nodes

            test = tools.create_tests(setup_func, file, **decorator_kwargs)[0]
            cluster = kwargs.get('cluster', None)
            if cluster is None:
                cluster = tools.system(lbann)
            weekly = kwargs.get('weekly', False)
            test(cluster, weekly, False)

        return wrapped

    return internal_tester


@dataclass
class ModelTester:
    """
    An object that is constructed within an ``lbann_test``-wrapped unit test.
    """

    # Input tensor (required for test to construct)
    input_tensor: Optional[Any] = None

    reference: Optional[lbann.Layer] = None  #: Reference LBANN node (optional)
    reference_tensor: Optional[
        Any] = None  #: Optional reference tensor to compare with

    # Tensor that will be used as the model objective function when checking
    # gradients. Required if check_gradients is True in tester.
    check_gradients_tensor: Optional[lbann.Layer] = None

    loss: Optional[lbann.Layer] = None  # Optional loss test
    tolerance: float = 0.0  #: Tolerance value for loss test

    # Optional additional metrics to use in test
    extra_metrics: List[lbann.BaseMetric] = field(default_factory=list)

    # Optional additional callbacks to use in test
    extra_callbacks: List[lbann.Callback] = field(default_factory=list)

    def inputs(self, tensor: Any) -> lbann.Layer:
        """
        Marks the given tensor as an input of the tested LBANN model, and
        returns a matching LBANN Input node (or a Slice/Reshape thereof).

        :param tensor: The input NumPy array to use.
        :return: An LBANN layer object that will serve as the input.
        """
        self.input_tensor = tensor
        inp = lbann.Input(data_field='samples')
        return slice_to_tensors(inp, tensor)

    def inputs_like(self, *tensors) -> List[lbann.Layer]:
        """
        Marks the given tensors as input of the tested LBANN model, and
        returns a list of matching LBANN Slice nodes, potentially reshaped to
        be like the input tensors.

        :param tensors: The input NumPy arrays to use.
        :return: A list of LBANN layer objects that will serve as the inputs.
        """
        minibatch_size = tensors[0].shape[0]  # Assume the first dimension

        # All tensors concatenated on the non-batch dimension
        all_tensors_combined = np.concatenate(
            [t.reshape(minibatch_size, -1) for t in tensors], axis=1)

        self.input_tensor = all_tensors_combined
        x = lbann.Input(data_field='samples')
        return slice_to_tensors(x, *tensors)

    def make_reference(self, ref: Any) -> lbann.Input:
        """
        Marks the given tensor as a reference output of the tested LBANN model,
        and returns a matching LBANN node.

        :param ref: The reference NumPy array to use.
        :return: An LBANN layer object that will serve as the reference.
        """
        # The reference is the second part of the input "samples"
        refnode = lbann.Input(data_field='samples')
        if self.input_tensor is None:
            raise ValueError('Please call ``inputs`` or ``inputs_like`` prior '
                             'to calling ``make_reference`` for correctness.')
        mbsize = self.input_tensor.shape[0]

        # Obtain reference
        refnode = lbann.Reshape(lbann.Identity(
            lbann.Slice(
                refnode,
                slice_points=[
                    numel(self.input_tensor) // mbsize,
                    (numel(self.input_tensor) + numel(ref)) // mbsize
                ],
            )),
                                dims=ref.shape[1:])

        # Store reference
        self.reference = refnode
        self.reference_tensor = ref
        return self.reference

    def set_check_gradients_tensor(self, tensor: lbann.Layer):
        """
        Sets the tensor to be used as the objective function when running the
        check gradients callback. When provided a non-scalar tensor, the
        objective function is the mean of the tensor.
        """
        self.check_gradients_tensor = lbann.Reduction(tensor, mode='mean')

    def set_loss_function(self,
                          func: Callable[[lbann.Layer, lbann.Layer],
                                         lbann.Layer],
                          output: lbann.Layer,
                          tolerance=None):
        """
        Sets a loss function and the LBANN test output to be measured for the
        test.
        This assumes that the first argument has two parameters (e.g.,
        ``MeanSquaredError``), where the first argument will be used for the
        LBANN output and the second will be used for the reference.

        :param func: The loss function.
        :param output: The LBANN model output to use.
        :param tolerance: Optional tolerance to set for the test. If ``None``,
                          the default tolerance of ``8*eps*mean(reference)``
                          will be used.
        """
        return self.set_loss(func(output, self.reference), tolerance)

    def set_loss(self,
                 loss: lbann.Layer,
                 tolerance: Optional[float] = None) -> None:
        """
        Sets an LBANN node to be measured for the test.

        :param loss: The LBANN graph node to use for the test.
        :param tolerance: Optional tolerance to set for the test. If ``None``,
                          the default tolerance of ``8*eps*mean(reference)``
                          will be used.
        """
        # Set loss node
        self.loss = loss

        # Set tolerance
        if tolerance is not None:
            self.tolerance = tolerance
        else:
            if self.reference_tensor is None:
                raise ValueError(
                    'Cannot set tolerance on loss function automatically '
                    'without a reference tensor. Either set tolerance '
                    'explicitly or call ``ModelTester.make_reference``.')
            # Default tolerance
            self.tolerance = abs(8 * np.mean(self.reference_tensor) *
                                 np.finfo(self.reference_tensor.dtype).eps)


def slice_to_tensors(x: lbann.Layer, *tensors) -> List[lbann.Layer]:
    """
    Slices an LBANN layer into multiple tensors that match the dimensions of
    the given numpy arrays.
    """
    slice_points = [0]
    offset = 0
    for tensor in tensors:
        offset += numel(tensor) // tensor.shape[0]

        slice_points.append(offset)
    lslice = lbann.Slice(x, slice_points=slice_points)
    return [
        lbann.Reshape(_ensure_bp(t, lbann.Identity(lslice)), dims=t.shape[1:])
        for t in tensors
    ]


def numel(tensor) -> int:
    """
    Returns the number of elements in a NumPy array, PyTorch array, or integer.
    """
    if isinstance(tensor, int):  # Integer
        return tensor
    elif hasattr(tensor, 'numel'):  # PyTorch array
        return tensor.numel()
    else:  # NumPy array
        return tensor.size


# Mimics the other tester's determination of working directory
def _get_work_dir(test_file: str) -> str:
    test_fname = os.path.realpath(test_file)
    # Create test name by removing '.py' from file name
    test_fname = os.path.splitext(os.path.basename(test_fname))[0]
    if not re.match('^test_.', test_fname):
        # Make sure test name is prefixed with 'test_'
        test_fname = 'test_' + test_fname
    return os.path.join(os.path.dirname(test_file), 'experiments', test_fname)


# Ensures that backpropagation would be run through the entire model
def _ensure_bp(tensor: Any, node: lbann.Layer) -> lbann.Sum:
    # Note: Sum with a weights layer so that gradient checking will
    # verify that error signals are correct.
    x_weights = lbann.Weights(initializer=lbann.ConstantInitializer(value=0.0))
    return lbann.Sum(
        node,
        lbann.WeightsLayer(
            weights=x_weights,
            dims=[numel(tensor) // tensor.shape[0]],
        ))
