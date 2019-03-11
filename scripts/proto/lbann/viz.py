"""Visualize LBANN models."""
from re import sub
from enum import Enum
from graphviz import Digraph
from lbann.proto import lbann_pb2, Model

def visualize_layer_graph(model, filename,
                          file_format='pdf',
                          label_format='name only',
                          graphviz_engine='dot'):
  """Visualize a model's layer graph and save to file.

  Args:
    model (`lbann_pb2.Model` or `lbann.proto.Model`): Neural network
      model.
    filename (`str`): Output file.
    file_format (`str`): Output file format.
    label_format (`str`): Displayed layer information (options:
      type-only, name-only, type-and-name, full).
    graphviz_engine (`str`): Graphviz visualization scheme.

  """

  # Get protobuf message
  if isinstance(model, lbann_pb2.Model):
    proto = model
  elif isinstance(model, Model):
    proto = model.export_proto()
  else:
    raise TypeError('expected `model` to be an '
                    '`lbann_pb2.Model` or a `lbann.proto.Model`')

  # Strip extension from filename
  if filename.endswith('.' + file_format):
    filename = filename[:-len(file_format)-1]

  # Convert label format to lowercase with no spaces
  label_format = sub(r' |-|_', '', label_format.lower())

  # Construct graphviz graph
  graph = Digraph(filename=filename, format=file_format, engine=graphviz_engine)
  graph.attr('node', shape='rect')

  # Construct nodes in layer graph
  layer_types = (set(lbann_pb2.Layer.DESCRIPTOR.fields_by_name.keys())
                 - set(['name', 'parents', 'children',
                        'data_layout', 'device_allocation', 'weights',
                        'num_neurons_from_data_reader', 'freeze',
                        'hint_layer', 'weights_data',
                        'top', 'bottom', 'type', 'motif_layer']))
  for l in proto.layer:

    # Determine layer type
    for type in layer_types:
      if l.HasField(type):
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
  for l in proto.layer:
    edges.update([(p, l.name) for p in l.parents.split()])
    edges.update([(l.name, c) for c in l.children.split()])
  graph.edges(edges)

  # Save to file
  graph.render(filename=filename, cleanup=True, format=file_format)
