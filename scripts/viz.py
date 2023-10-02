#!/usr/bin/env python3
"""Visualize an LBANN model's layer graph and save to file."""

import argparse
import re
import graphviz
import google.protobuf.text_format
from lbann import lbann_pb2, layers_pb2

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
                    default='graph.pdf',
                    type=str,
                    help='output file (default: graph.pdf)')
parser.add_argument('--file-format',
                    action='store',
                    default='pdf',
                    type=str,
                    help='output file format (default: pdf)',
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
args = parser.parse_args()

# Strip extension from filename
filename = args.output
file_format = args.file_format
if filename.endswith('.' + file_format):
    filename = filename[:-len(file_format) - 1]

# Convert label format to lowercase with no spaces
label_format = re.sub(r' |-|_', '', args.label_format.lower())

# Read prototext file
proto = lbann_pb2.LbannPB()
with open(args.input, 'r') as f:
    google.protobuf.text_format.Merge(f.read(), proto)
model = proto.model

# Construct graphviz graph
graph = graphviz.Digraph(filename=filename,
                         format=file_format,
                         engine=args.graphviz_engine)
graph.attr('node', shape='rect')

# Construct nodes in layer graph
layer_types = (set(layers_pb2.Layer.DESCRIPTOR.fields_by_name.keys()) - set([
    'name', 'parents', 'children', 'datatype', 'data_layout',
    'device_allocation', 'weights', 'freeze', 'hint_layer', 'top', 'bottom',
    'type', 'motif_layer'
]))
for l in model.layer:

    # Determine layer type
    type = ''
    for _type in layer_types:
        if l.HasField(_type):
            type = getattr(l, _type).DESCRIPTOR.name
            break

    # Construct node label
    label = ''
    if label_format == 'nameonly':
        label = l.name
    elif label_format == 'typeonly':
        label = type
    elif label_format == 'typeandname':
        label = '<{0}<br/>{1}>'.format(type, l.name)
    elif label_format == 'full':
        label = '<'
        for (index, line) in enumerate(str(l).strip().split('\n')):
            if index > 0:
                label += '<br/>'
            label += line
        label += '>'

    # Add layer as layer graph node
    graph.node(l.name, label=label)

# Add parent/child relationships as layer graph edges
edges = set()
for l in model.layer:
    edges.update([(p, l.name) for p in l.parents.split()])
    edges.update([(l.name, c) for c in l.children.split()])
graph.edges(edges)

# Save to file
graph.render(filename=filename, cleanup=True, format=file_format)
