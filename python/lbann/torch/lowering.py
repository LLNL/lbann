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
Functionality for using PyTorch Dynamo to selectively lower PyTorch graphs
and maintain neural network semantics necessary for LBANN.
"""

import lbann
from lbann.torch import converters, helpers, opaque

import torch
from torch import fx
import torch._inductor.compile_fx as inductor
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._subclasses.fake_tensor import FakeTensorMode
from typing import Any, Dict, List, Optional
import warnings


def dynamo_callback(input_names: List[str], with_weights: bool,
                    graph_module: fx.GraphModule,
                    example_inputs: List[torch.Tensor]) -> helpers.LBANNGraph:
    """
    Entry point into the LBANN frontend from the PyTorch Dynamo interface.
    Converts the Dynamo-parsed ``torch.fx`` ``GraphModule`` to an LBANN
    list of Layers (constituting a graph). Note that this function never returns
    but instead raises the resulting graph as an exception, in order to work
    around the PyTorch framework expecting certain (callable) return values.

    :param input_names: Names of the parameters to use as inputs.
    :param with_weights: If True (default), also stores the parameters of
                         the given graph as constant initializers of the LBANN
                         graph's weights.
    :param graph_module: Dynamo-compiled FX graph module.
    :param example_inputs: Example input tensors fed at compile time (used for
                           shape and type inference).
    :return: An LBANNGraph object containing the LBANN layers.
    """

    # Verbose printout
    # graph_module.graph.print_tabular()

    # Call internal function that parses the graph
    _, inputs, _ = replace_fx(graph_module, example_inputs, with_weights)

    # Reconstruct LBANN graph and exit with original graph
    if not input_names:
        layer_graph = list(lbann.traverse_layer_graph(inputs))
    else:
        layer_graph = list(
            lbann.traverse_layer_graph(inputs[-len(input_names):]))

    raise helpers.LBANNGraph(layer_graph)


def replace_fx(gm: fx.GraphModule,
               example_inputs: List[torch.Tensor],
               with_weights: bool,
               replaced: Optional[Dict[fx.Node, lbann.Layer]] = None,
               subgraph: bool = False):
    """
    Converts a PyTorch FX ``GraphModule`` to an LBANN graph in a 4-pass manner.
    First pass ensures all tensors have propagated types and shapes (using both
    PyTorch Dynamo and Opaque modules registered in LBANN). The second pass then
    selectively lowers every graph node that has no replacement (see 
    ``partial_lowering``), followed by re-propagating types/shapes. Lastly, a
    fourth pass converts every remaining function and module to the matching
    LBANN subgraphs using the converters in ``replacements``.

    :param gm: The graph module to convert.
    :param example_inputs: Example input tensors fed at compile time (used for
                           shape and type inference).
    :param with_weights: If True (default), also stores the parameters of
                         the given graph as constant initializers of the LBANN
                         graph's weights.
    :param replaced: Internal field used for recursive parsing. Maintains
                     a dictionary of already-performed replacements.
    :param subgraph: Internal field used for recursive parsing. Set to True if
                     a subgraph is being replaced.
    """

    # Tensor metadata (type, shape, etc.) propagation
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    FakeTensorProp(gm, fake_mode).propagate(*example_inputs)

    # Partially lower graph when a module or function is unsupported
    if not subgraph:
        gm = partial_lowering(gm)
        gm.graph.eliminate_dead_code()

    # Re-propagate tensor metadata after lowering
    FakeTensorProp(gm, fake_mode).propagate(*example_inputs)

    replaced = replaced or {}
    inputs = []
    outputs = []
    rep = set()

    # Helper function that replaces sequences and objects (for handling
    # arguments)
    def repl(args):
        if not isinstance(args, (tuple, list)):
            return replaced.get(args, args)

        result = []
        for arg in args:
            if isinstance(arg, (tuple, list)):
                result.append(repl(arg))
            else:
                try:
                    result.append(replaced.get(arg, arg))
                except:
                    result.append(arg)
        return result

    # Traverse FX graph
    for node in gm.graph.nodes:
        # Replace once
        if node in replaced:
            continue

        if node.op == "call_module":
            submodule = _fetch_attr(gm, node.target)

            # Extract original module if opaque
            if isinstance(submodule, opaque.Opaque):
                submodule = submodule.origmod

            if isinstance(submodule, fx.GraphModule):
                raise NotImplementedError('Need to inline nested graph module')
            elif type(submodule) in converters.modules:
                # If module is replaceable, replace it
                node.stack_trace = f'{type(submodule).__name__}_{len(replaced)}'
                replaced[node] = converters.modules[type(submodule)](
                    submodule, *repl(node.args),
                    **{k: repl(v)
                       for k, v in node.kwargs.items()})

                # Convert weights
                if with_weights:
                    if type(submodule) not in converters.module_parameters:
                        if len(list(submodule.parameters())) > 0:
                            warnings.warn('No converter found for weights of '
                                          f'module type "{type(submodule)}"!')
                    else:
                        converters.module_parameters[type(submodule)](
                            submodule, replaced[node])

                replaced[node].name = node.name
                replaced[node].shape = node.meta['val'].shape
                rep.add(node.stack_trace)
            else:
                raise NameError(
                    'Could not find high-level replacement for '
                    f'module type "{type(submodule)}" ({funcname}). Lowering')

        elif node.op == "call_function":
            funcname = str(node.target)
            if funcname not in converters.functions:
                # Try by qualified (module and function) name
                funcname = node.target.__module__ + '.' + node.target.__name__

            if funcname in converters.functions:
                node.stack_trace = f'{funcname}_{len(replaced)}'
                replaced[node] = converters.functions[funcname](*repl(
                    node.args), **{k: repl(v)
                                   for k, v in node.kwargs.items()})
                replaced[node].name = node.name
                replaced[node].shape = node.meta['val'].shape
                rep.add(node.stack_trace)
            else:  # Already in lowered subgraph, all low-level replacements must be supported
                raise NameError(
                    f'Function "{str(node.target)}" ({funcname}) is unsupported.'
                )

        # Special case for torch.unbind
        elif node.op == 'call_method' and str(node.target) == 'unbind':
            # Get slice points
            shp = node.args[0].meta['val'].shape
            s = shp[node.args[1]]
            replaced[node] = lbann.Slice(replaced[node.args[0]],
                                         axis=node.args[1],
                                         slice_points=list(range(s)))
            # Loop over children in-order to produce the slices
            for user in node.users.keys():
                replaced[user] = lbann.Identity(replaced[node])
                replaced[user].shape = user.meta['val'].shape
        # Special case for torch.split
        elif node.op == 'call_method' and str(node.target) == 'split':
            # Get slice points
            shp = node.args[0].meta['val'].shape
            dim = node.kwargs['dim']
            slice_size_or_slices = node.args[1]
            if isinstance(slice_size_or_slices, list):
                ind = 0
                slice_points = [0]
                for s in slice_size_or_slices:
                    ind += s
                    slice_points.append(s)
            else:  # Single number: slice size (last one cut off)
                num_even_slices = shp[dim] // slice_size_or_slices
                ind = 0
                slice_points = [0]
                for _ in range(num_even_slices):
                    ind += slice_size_or_slices
                    slice_points.append(ind)
                # Add final slice
                if num_even_slices != len(node.users):
                    slice_points.append(shp[dim])

            replaced[node] = lbann.Slice(replaced[node.args[0]],
                                         axis=dim,
                                         slice_points=slice_points)
            # Loop over children in-order to produce the slices
            for user in node.users.keys():
                replaced[user] = lbann.Identity(replaced[node])
                replaced[user].shape = user.meta['val'].shape

        elif node.op == "placeholder" and not subgraph:
            lbann_node = lbann.Input(data_field='samples')
            # If shape is more than two-dimensional, add a Reshape node
            if len(node.meta['val'].shape) > 2:
                lbann_node = lbann.Reshape(lbann_node,
                                           dims=node.meta['val'].shape[1:])
            replaced[node] = lbann_node
            inputs.append(replaced[node])
            replaced[node].shape = node.meta['val'].shape
        elif node.op == "get_attr" and not subgraph:
            # Could be a parameter
            attr = _fetch_attr(gm, node.target)
            if isinstance(attr, torch.nn.Parameter):  # Setup Weights layer
                if with_weights:
                    lbann_node = lbann.WeightsLayer(
                        dims=attr.shape,
                        weights=lbann.Weights(
                            initializer=lbann.ValueInitializer(
                                values=attr.detach().cpu().numpy().flat)))
                else:
                    lbann_node = lbann.WeightsLayer(dims=attr.shape)
                make_input = False
            else:  # Unknown, assume input
                lbann_node = lbann.Input(data_field='samples')
                if len(node.meta['val'].shape) > 2:
                    lbann_node = lbann.Reshape(lbann_node,
                                               dims=node.meta['val'].shape)
                make_input = True

            replaced[node] = lbann_node
            replaced[node].shape = node.meta['val'].shape
            if make_input:
                inputs.append(replaced[node])
        elif node.op == "output":
            outputs = repl(node.args)

    return replaced, inputs, outputs


# Adapted from PyTorch source - recursively fetches attributes from target name.
def _fetch_attr(x: Any, target: str):
    target_atoms = target.split('.')
    attr_itr = x
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError('Node referenced nonexistent target '
                               f'{".".join(target_atoms[:i])}')
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def lower_single_node(gm: fx.GraphModule, node: fx.Node) -> fx.GraphModule:
    """
    Selectively lowers a single ``torch.fx`` graph node into a graph module
    of its own. This is used for functions that have no replacements (for
    example, an esoteric activation function). Uses PyTorch Inductor to perform
    the lowering and returns another graph module.

    :param gm: Original graph module.
    :param node: Node to lower.
    :return: A new FX graph module containing the lowered graph.
    """

    # Make graph with one node
    subgraph = fx.Graph()
    value_remap = {}

    # Inputs
    args = []
    input_nodes = []
    for input in node.all_input_nodes:
        value_remap[input] = subgraph.placeholder(input.name)
        input_nodes.append(input)
        args.append(input.meta['val'])

    # Computation
    output = subgraph.node_copy(node, lambda n: value_remap[n])

    # Output
    subgraph.output((output, ))

    # Make and lower the graph module
    subgm = fx.GraphModule(gm, subgraph)

    # Get graph from internal function without modifying PyTorch
    class GetLoweredGraph(Exception):

        def __init__(self, graph, inputs) -> None:
            self.graph = graph
            self.inputs = inputs

    def lower_only(lowered: fx.GraphModule,
                   example_params_and_inputs: List[torch.Tensor],
                   num_fixed=0,
                   is_backward=False,
                   is_inference=False,
                   **kwargs):
        if is_backward:
            return
        if num_fixed > 0:
            raise ValueError(
                'Cannot convert parameterized module. Please register a '
                'module replacement.')
        raise GetLoweredGraph(lowered, example_params_and_inputs)

    def lower():
        try:
            inductor.compile_fx(
                subgm,
                args,
                inner_compile=lower_only,
            )
        except GetLoweredGraph as ex:
            return ex.graph, ex.inputs

        # If no internal exception was raised, ``lower_only`` was not called
        # properly
        raise ValueError('Could not obtain lowered graph')

    # End of PyTorch inductor lowering plumbing

    lowered_graph, _ = lower()

    return lowered_graph


def partial_lowering(gm: fx.GraphModule) -> fx.GraphModule:
    """
    A pass on a PyTorch FX graph module that finds unsupported operations (i.e.,
    no registered converter) and selectively lowers them using
    ``lower_single_node``.

    :param gm: The input graph module.
    :return: A graph module where unsupported operations are expanded to new
             subgraphs.
    """

    new_graph = fx.Graph()
    env = {}
    tracer = fx.proxy.GraphAppendingTracer(new_graph)
    skip_nodes = set()

    # Traverse graph in-order
    for node in gm.graph.nodes:
        # Decide whether this node should be lowered
        decompose = False
        if node not in skip_nodes:
            if node.op == 'call_module':
                submodule = _fetch_attr(gm, node.target)

                # Extract original module if opaque
                if isinstance(submodule, opaque.Opaque):
                    submodule = submodule.origmod

                if isinstance(submodule, fx.GraphModule) or type(
                        submodule) not in converters.modules:
                    decompose = True
                    warnings.warn(
                        'Could not find replacement for '
                        f'module type "{type(submodule)}". Lowering.')
            elif node.op == 'call_function':
                funcname = str(node.target)
                if funcname not in converters.functions:
                    # Try by qualified (module and function) name
                    funcname = node.target.__module__ + '.' + node.target.__name__
                if funcname not in converters.functions:
                    decompose = True
                    warnings.warn(
                        'Could not find replacement for '
                        f'function "{str(node.target)}" ({funcname}). Lowering.'
                    )
            elif node.op == 'call_method':
                decompose = True
                warnings.warn(
                    f'Lowering method "{str(node.target)}" operating '
                    f'on object {str(node.args[0])}.')

        if node.op == 'call_method' and str(
                node.target) in ('unbind', 'split'):
            # Special case for unbind + getitem
            assert len(node.users) == len(node.meta['val'])
            for user in node.users.keys():
                assert (user.op == 'call_function'
                        and str(user.target) == '<built-in function getitem>')
                skip_nodes.add(user)
            decompose = False
        # End of lowering decision

        if decompose:
            # Lower and reconstruct graph
            lowered = lower_single_node(gm, node)
            proxy_args = [
                fx.Proxy(env[x.name], tracer) if isinstance(x, fx.Node) else x
                for x in node.args if isinstance(x, fx.Node)
            ]
            output_proxy = lowered(*proxy_args)
            new_node = output_proxy[0].node
            env[node.name] = new_node
        else:
            # Copy node as-is
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node

    return fx.GraphModule(gm, new_graph)
