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
""" Python interface to evaluate a single set of inputs. """

import copy
import lbann
from lbann.launcher import make_timestamped_work_dir
from lbann.contrib import single_tensor_data_reader
import numpy as np
import numpy.typing as npt
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings


def evaluate(
    model: Union[lbann.Model, List[lbann.Layer]],
    inputs: npt.NDArray,
    outputs: Optional[List[str]] = None,
    **kwargs,
) -> Union[npt.NDArray, Tuple[npt.NDArray]]:
    """
    Runs a given LBANN model once on the given inputs and returns its forward
    propagation outputs. All tensors are numpy arrays.

    :param model: The LBANN model or layer graph to evaluate.
    :param inputs: A tensor with the inputs to the model. It will be mapped
                   to the ``lbann.Input`` with the ``samples`` data field.
    :param outputs: An optional list of layer names to output as the return
                    value. If not given, returns all layers without children.
    :param kwargs: Additional keyword arguments to pass onto ``lbann.run``
    :return: Output tensor or tensors of the LBANN model.
    """

    ########################
    # Canonicalize arguments

    # Set model to always be an lbann.Model
    if isinstance(model, (list, tuple, set, lbann.Layer)):
        if isinstance(model, lbann.Layer):
            model = [model]
        model = lbann.Model(0, model)

    # Obtain outputs if not given
    if not outputs:
        outputs = [l.name for l in model.layers if not l.children]
    #####################
    if 'job_name' not in kwargs:
        kwargs['job_name'] = 'lbann_evaluate'

    workdir = make_timestamped_work_dir(**kwargs)

    # Reset fields for evaluation
    old_epochs = model.epochs
    old_callbacks = model.callbacks
    old_metrics = model.metrics

    try:
        model.epochs = 0
        model.callbacks = [
            lbann.CallbackDumpOutputs(batch_interval=1,
                                      execution_modes='test',
                                      directory=workdir,
                                      layers=' '.join(outputs))
        ]
        model.metrics = []

        data_reader = _setup_data_reader(inputs, workdir)
        trainer = lbann.Trainer(inputs.shape[0])

        ########################
        # Run model
        lbann.run(trainer, model, data_reader, lbann.NoOptimizer(), workdir,
                  **kwargs)

        #######################
        # Collect outputs
        output_tensors = _collect_outputs(outputs, workdir, inputs.dtype)

    finally:
        # Set fields back to original state
        model.epochs = old_epochs
        model.callbacks = old_callbacks
        model.metrics = old_metrics

    # Unpack tuple if single-element
    if len(output_tensors) == 1:
        output_tensors = output_tensors[0]

    return output_tensors


def _setup_data_reader(inputs: npt.NDArray, workdir: str):
    # Save inputs
    if len(inputs.shape) == 1:  # Minibatch dimension must exist
        inputs = inputs.reshape(1, inputs.shape[0])
    elif len(inputs.shape) > 2:
        warnings.warn(
            f'{len(inputs.shape)}-dimensional tensor given, all dimensions '
            'beyond the second one will be flattened')
        inputs = inputs.reshape(inputs.shape[0], -1)

    np.save(os.path.join(workdir, 'data.npy'), inputs)

    # Construct protobuf message for data reader
    file_name = os.path.realpath(single_tensor_data_reader.__file__)
    dir_name = os.path.dirname(file_name)
    module_name = os.path.splitext(os.path.basename(file_name))[0]
    reader = lbann.reader_pb2.Reader()
    reader.name = 'python'
    reader.role = 'test'
    reader.shuffle = False
    reader.percent_of_data_to_use = 1.0
    reader.python.module = module_name
    reader.python.module_dir = dir_name
    reader.python.sample_function = 'get_sample'
    reader.python.num_samples_function = 'num_samples'
    reader.python.sample_dims_function = 'sample_dims'
    train_reader = copy.deepcopy(reader)
    train_reader.role = 'train'

    data_reader = lbann.reader_pb2.DataReader()
    data_reader.reader.extend([train_reader, reader])

    return data_reader


def _collect_outputs(output_names: List[str], workdir: str,
                     dtype: np.dtype) -> Tuple[npt.NDArray]:
    output_dir = os.path.join(workdir, 'trainer0', 'model0')
    file_prefix = 'sgd.testing.epoch.0.step.0'

    outputs = []
    for name in output_names:
        fname = os.path.join(output_dir, f'{file_prefix}_{name}_output0.csv')
        outputs.append(np.loadtxt(fname, dtype, delimiter=','))

    return tuple(outputs)
