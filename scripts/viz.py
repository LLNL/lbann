#!/usr/bin/env python3
import argparse
import google.protobuf.text_format as txtf
from lbann.proto import lbann_pb2
import lbann.viz

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Visualize layer graph for LBANN model.')
parser.add_argument(
    'input', action='store', type=str,
    help='model prototext file')
parser.add_argument(
    'output', action='store', nargs='?',
    default='graph.pdf', type=str,
    help='output file (default: graph.pdf)')
parser.add_argument(
    '--file-format', action='store', default='pdf', type=str,
    help='output file format (default: pdf)', metavar='FORMAT')
parser.add_argument(
    '--label-format',
    action='store', default='type-only', type=str,
    choices=('type-only', 'name-only', 'type-and-name', 'full'),
    help='displayed layer info (default: type-only)')
parser.add_argument(
    '--graphviz-engine', action='store', default='dot', type=str,
    help='Graphviz visualization scheme (default: dot)', metavar='ENGINE')
args = parser.parse_args()

# Parse prototext file
proto = lbann_pb2.LbannPB()
with open(args.input, 'r') as f:
    txtf.Merge(f.read(), proto)

# Visualize
lbann.viz.visualize_layer_graph(proto.model, args.output,
                                file_format=args.file_format,
                                label_format=args.label_format,
                                graphviz_engine=args.graphviz_engine)
