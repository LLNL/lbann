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
"""
Provides an API for LBANN graph compilation with PyTorch 2.x.
"""

# Version check
try:
    import torch
    if int(torch.__version__.split('.')[0]) < 2:
        raise ImportError(
            'PyTorch 2.0 or newer is required for the LBANN torch frontend. '
            f'Found version {torch.__version__}')
except (ModuleNotFoundError, ImportError):
    raise ImportError(
        'PyTorch (version 2.0 or newer) is required for the LBANN torch '
        'frontend.')
# End of version check

import functools
import inspect
import lbann
from lbann.torch import converters, opaque, lowering
from lbann.torch.helpers import LBANNGraph
from torch import nn, _dynamo as dynamo
from torch._dynamo.exc import BackendCompilerFailed
from typing import Any, Callable, List, Optional, Union


def compile(module_or_function: Union[nn.Module, Callable[..., Any]],
            *sample_args,
            trace: bool = False,
            **sample_kwargs) -> List[lbann.Layer]:
    """
    Compiles the given PyTorch module or function into an LBANN graph.
    Internally uses PyTorch Dynamo to parse the module/function's contents
    and the module and function converters in ``lbann.torch.replacements``.
    A ``trace`` mode is also available to work with dynamic modules that
    Dynamo currently cannot compile into one graph.

    :param module_or_function: ``torch.nn.Module`` or function to compile.
    :param sample_args: Positional arguments to pass in for compilation (used
                        only if tracing is enabled).
    :param trace: If False (default), compiles the function statically (e.g.,
                  supporting conditions). Otherwise, traces through the function
                  in order to construct the LBANN graph.
    :param sample_kwargs: Named arguments to pass in for compilation. Note that
                          this is necessary to compile ahead-of-time (without
                          tracing).
    :return: An LBANN graph given as a list of Layers.
    """
    if not trace and sample_args:
        raise ValueError('All arguments must be named in ahead-of-time '
                         'compilation')

    if trace:
        return _trace(module_or_function, sample_args, sample_kwargs)

    cmod = lazy_compile(module_or_function)
    return cmod(*tuple(sample_kwargs.values()))


def _trace(f, example_args, example_kwargs) -> List[lbann.Layer]:
    """
    Traces through a function or Module to obtain an LBANN graph.

    :param f: Function or module to trace.
    :param example_args: Positional arguments to pass in for tracing.
    :param example_kwargs: Keyword arguments to pass in for tracing.
    :return: An LBANN graph given as a list of Layers.
    """
    converters.load_replacements()

    huggingface = False
    if isinstance(f, nn.Module):
        # Workaround to automatically call Huggingface's tracer
        for m in f.modules():
            if m.__module__.startswith('transformers'):
                huggingface = True
                break

    if huggingface:
        from transformers.utils import fx
        input_names = [k for k, v in example_kwargs.items() if v is not None]
        input_vals = [example_kwargs[k] for k in input_names]
        g = fx.symbolic_trace(f, input_names=input_names)
        try:
            lowering.dynamo_callback(input_names, g, input_vals)
        except LBANNGraph as ex:
            return ex.graph
    else:
        # Otherwise, call traced function with the standard tracer
        from torch.fx.experimental import proxy_tensor
        fxf = proxy_tensor.make_fx(f, tracing_mode='real')
        g = fxf(*example_args, *tuple(example_kwargs.values()))
        try:
            lowering.dynamo_callback([], g, example_args)
        except LBANNGraph as ex:
            return ex.graph

    raise RuntimeError(
        'Could not extract an LBANN graph while tracing the given PyTorch '
        'module or function')


def _get_argnames(f) -> List[str]:
    """
    Returns the argument names of a Python function.
    """
    try:
        return list(inspect.signature(f).parameters.keys())
    except AttributeError:
        return inspect.getargspec(f).args


def _get_module_argnames(f: Union[nn.Module, Callable[..., Any]]) -> List[str]:
    """
    Returns the argument names of a Python function or ``nn.Module``.
    """
    if isinstance(f, nn.Module):
        return _get_argnames(f.forward)
    return _get_argnames(f)


def lazy_compile(module_or_function: Union[nn.Module, Callable[..., Any]]):
    """
    Compile a Python function with PyTorch to an LBANN graph lazily. This means
    that whenever the decorated function is called, an LBANN graph will be
    returned instead of calling the function.

    :param module_or_function: ``torch.nn.Module`` or function to mark for
                               compilation.
    :param argnames: A list of strings representing the argument names. If not
                     given, tries to automatically obtain them from the
                     function's signature.
    """
    converters.load_replacements()

    argnames = _get_module_argnames(module_or_function)

    f = module_or_function
    if isinstance(f, nn.Module):
        f = opaque.wrap_modules(f)
        print('Replaced opaque operators on the module tree:',
              opaque.count_replaced_submodules(f))

    f = dynamo.optimize(functools.partial(lowering.dynamo_callback, argnames),
                        nopython=True)(f)

    @functools.wraps(f)
    def _wrapped(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except BackendCompilerFailed as ex:
            if isinstance(ex.inner_exception, LBANNGraph):
                return ex.inner_exception.graph
            raise
        except LBANNGraph as ex:
            return ex.graph

        raise RuntimeError(
            'Could not extract an LBANN graph from PyTorch module '
            'or function')

    return _wrapped
