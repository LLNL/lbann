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
Conversion utilities from PyTorch ``nn.Module``s and built-in functions
to LBANN Layers and Weights.
"""
import lbann
import torch
from torch import nn
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

# Mapping of PyTorch modules to LBANN Layer generator functions. The functions
# in the dictionary values accept the module and any parameters it was called
# with.
modules: Dict[Type[nn.Module], Callable[..., lbann.Layer]] = {}

# Mapping of PyTorch built-in functions to LBANN Layer generator functions.
# Dictionary keys store functions by their name, and the functions
# in the dictionary values accept the same parameters as the function would.
functions: Dict[str, Callable[..., Any]] = {}

# Mapping of PyTorch module types to LBANN parameter generator functions.
# The functions in the dictionary values accept the module object and the
# matching LBANN layer, and set its weights after performing the necessary
# transformations (e.g., transposition).
module_parameters: Dict[Type[nn.Module], Callable[[nn.Module, lbann.Layer],
                                                  None]] = {}

# Mapping of PyTorch module types to shape inference functions. Shape inference
# functions accept the ``nn.Module`` objects along with any given parameters
# and return a tuple of the resulting tensor's dtype, shape, and device.
# Used for "skipping" certain layers that cannot be parsed by PyTorch Dynamo
# directly (such as GNNs).
opaque_shape_inference: Dict[Type[nn.Module],
                             Tuple[str, Callable[...,
                                                 Tuple[torch.dtype, Tuple[int],
                                                       torch.device]]]] = {}


def load_replacements():
    """
    Populates the replacement dictionaries
    """
    # Import modules to populate the above dictionaries
    from lbann.torch.replacements import functions, modules


def register_function(layername: str):
    """
    
    Syntax::
    
        @register_function("")
        def replacement():
            return lbann.MatchingClass(...)

    """

    def func(f):
        functions[layername] = f
        return f

    return func


def register_module(modclass: Union[Type[nn.Module],
                                    Sequence[Type[nn.Module]]]):
    """
    
    Syntax::
    
        @register_module(ClassName)
        def replacement(mod: ClassName):
            return lbann.MatchingClass(...)

    """

    def func(f):
        if isinstance(modclass, (list, tuple)):
            for cls in modclass:
                modules[cls] = f
        else:
            modules[modclass] = f

        return f

    return func


def register_opaque_shape_inference(
        modclass: Union[Type[nn.Module], Sequence[Type[nn.Module]]]):
    """

    Syntax::

        @register_opaque_shape_inference(GCNConv)
        def infer(mod: GCNConv, *args, **kwargs):
            return dtype, shape, device

    """

    def func(f):
        if isinstance(modclass, (list, tuple)):
            for cls in modclass:
                opaque_shape_inference[cls] = (cls.__name__, f)
        else:
            opaque_shape_inference[modclass] = (modclass.__name__, f)

        return f

    return func
