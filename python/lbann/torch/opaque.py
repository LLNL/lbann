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
Opaque PyTorch module functionality.
"""

import torch
import torch._dynamo as dynamo
import torch.nn
from typing import Any, Callable


@dynamo.allow_in_graph
class Opaque(torch.nn.Module):
    """
    An opaque PyTorch module that prohibits Dynamo from inlining the contents
    of the original module, which might not be compilable.
    """

    def __init__(self, name: str, origmod: torch.nn.Module,
                 shape_inference: Callable[..., Any]) -> None:
        super().__init__()
        self.name = name
        self.origmod = origmod  # The wrapped module
        self.shape_infer = shape_inference
        self.unique_key = id(self)

    def forward(self, *args, **kwargs):
        # Create an output tensor with some constant value but the right shape
        # and type, so that PyTorch can create a coherent graph
        dtype, shape, device = self.shape_infer(self.origmod, *args, **kwargs)
        result = torch.full(shape, self.unique_key, dtype=dtype, device=device)
        return result


def wrap_opaque_modules(mod: torch.nn.Module) -> torch.nn.Module:
    """
    Traverses a PyTorch module tree and wraps any module with custom shape
    inference with an ``Opaque`` class.

    :param mod: The module to traverse.
    :return: A new module where every relevant submodule (or the module itself)
             is replaced with an ``Opaque`` module.
    """
    # Avoid cyclical imports
    from lbann.torch.converters import opaque_shape_inference

    # Module itself should be opaque
    if type(mod) in opaque_shape_inference:
        newname, shape_infer = opaque_shape_inference[type(mod)]
        return Opaque(newname, mod, shape_infer)

    # Any submodule should be opaque
    for name, m in mod.named_children():
        if isinstance(m, Opaque):
            continue
        if type(m) in opaque_shape_inference:
            newname, shape_infer = opaque_shape_inference[type(m)]
            setattr(mod, name, Opaque(newname, m, shape_infer))
        elif m is not mod:
            wrap_opaque_modules(m)

    return mod


def count_replaced_submodules(mod: torch.nn.Module) -> int:
    """
    Helper function that counts how many modules or submodules were replaced.
    """
    if isinstance(mod, Opaque):
        return 1

    result = 0
    for _, m in mod.named_children():
        if isinstance(m, Opaque):
            result += 1
            continue
        if m is not mod:
            result += count_replaced_submodules(m)

    return result
