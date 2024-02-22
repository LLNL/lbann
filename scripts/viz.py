#!/usr/bin/env python3
"""Visualize an LBANN model's layer graph and save to file."""

import argparse
import random
import re
import graphviz
from lbann import lbann_pb2, layers_pb2
from lbann.proto import serialize

# Pastel rainbow (slightly shuffled) from colorkit.co
palette = [
    '#ffffff', '#a0c4ff', '#ffadad', '#fdffb6', '#caffbf', '#9bf6ff',
    '#bdb2ff', '#ffc6ff', '#ffd6a5'
]

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Visualize an LBANN model\'s layer graph and save to file.')
parser.add_argument('input',
                    action='store',
                    type=str,
                    help='model prototext file')
parser.add_argument('output',
                    action='store',
                    nargs='?',
                    default='graph.dot',
                    type=str,
                    help='output file (default: graph.dot)')
parser.add_argument('--file-format',
                    action='store',
                    default='dot',
                    type=str,
                    help='output file format (default: dot)',
                    metavar='FORMAT')
parser.add_argument('--label-format',
                    action='store',
                    default='type-only',
                    type=str,
                    choices=('type-only', 'name-only', 'type-and-name',
                             'full'),
                    help='displayed layer info (default: type-only)')
parser.add_argument('--graphviz-engine',
                    action='store',
                    default='dot',
                    type=str,
                    help='Graphviz visualization scheme (default: dot)',
                    metavar='ENGINE')
parser.add_argument('--color-cross-grid',
                    action='store_true',
                    default=False,
                    help='Highlight cross-grid edges')
args = parser.parse_args()

# Strip extension from filename
filename = args.output
file_format = args.file_format
if filename.endswith('.' + file_format):
    filename = filename[:-len(file_format) - 1]

# Convert label format to lowercase with no spaces
label_format = re.sub(r' |-|_', '', args.label_format.lower())

# Read prototext file
proto = serialize.generic_load(args.input)
model = proto.model

# Construct graphviz graph
graph = graphviz.Digraph(filename=filename,
                         format=file_format,
                         engine=args.graphviz_engine)
graph.attr('node', shape='rect')

layer_to_grid_tag = {}

# Construct nodes in layer graph
layer_types = (set(layers_pb2.Layer.DESCRIPTOR.fields_by_name.keys()) - set([
    'name', 'parents', 'children', 'datatype', 'data_layout',
    'device_allocation', 'weights', 'freeze', 'hint_layer', 'top', 'bottom',
    'type', 'motif_layer', 'parallel_strategy', 'grid_tag'
]))
for l in model.layer:

    # Determine layer type
    ltype = ''
    for _type in layer_types:
        if l.HasField(_type):
            ltype = getattr(l, _type).DESCRIPTOR.name
            break

    # If operator layer, use operator type
    if ltype == 'OperatorLayer':
        url = l.operator_layer.ops[0].parameters.type_url
        ltype = url[url.rfind('.') + 1:]

    # Construct node label
    label = ''
    if label_format == 'nameonly':
        label = l.name
    elif label_format == 'typeonly':
        label = ltype
    elif label_format == 'typeandname':
        label = '<{0}<br/>{1}>'.format(ltype, l.name)
    elif label_format == 'full':
        label = '<'
        for (index, line) in enumerate(str(l).strip().split('\n')):
            if index > 0:
                label += '<br/>'
            label += line
        label += '>'

    # Add layer as layer graph node
    tag = l.grid_tag.value
    layer_to_grid_tag[l.name] = tag
    attrs = {}
    if tag != 0:
        attrs = dict(style='filled', fillcolor=palette[tag % len(palette)])
    graph.node(l.name, label=label, **attrs)

# Add parent/child relationships as layer graph edges
edges = set()
cross_grid_edges = set()
for l in model.layer:
    tag = layer_to_grid_tag[l.name]
    for p in l.parents:
        if tag != layer_to_grid_tag[p]:
            cross_grid_edges.add((p, l.name))
        else:
            edges.add((p, l.name))

    for c in l.children:
        if tag != layer_to_grid_tag[c]:
            cross_grid_edges.add((l.name, c))
        else:
            edges.add((l.name, c))

graph.edges(edges)
if args.color_cross_grid:
    for u, v in cross_grid_edges:
        graph.edge(u, v, color='red')
else:
    graph.edges(cross_grid_edges)

# Save to file
graph.render(filename=filename, cleanup=True, format=file_format)
