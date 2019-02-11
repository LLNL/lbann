#!/usr/bin/env python3

from sys import *
import os
import os.path
import pprint
from graphviz import Digraph

from lbann.viz.layer import Layer
from lbann.viz.properties import Properties

def fixSequentialParents(layers) :
  '''a hack for models that don't contain parent and children fields'''
  num_layers_with_parents = 0
  num_layers_with_children = 0
  for layer in layers :
    if len(layer.parents()) != 0 : num_layers_with_parents += 1
    if len(layer.children()) != 0 : num_layers_with_children += 1
  if num_layers_with_parents == 0 :
    print()
    print('NOTE: this model does not appear to have any parent fields;')
    print('      dealing with that ...')
    print()
    assert(num_layers_with_children == 0)
    for j in range(1, len(layers)) :
      layers[j].setParents(layers[j-1])

#WARNING: this works for tim's rnn prototext, but may not generalize
def getLinkedLayers(layers) :
  r = []
  w = {}
  for layer in layers :
    my_name = layer.name()
    links = layer.linkedLayers()
    for x in links :
      if my_name == x :
        w[my_name] = set([my_name])
  for layer in layers :
    links = layer.linkedLayers()
    my_name = layer.name()
    for x in links :
      if my_name != x :
        if my_name in w :
          w[my_name].add(x)
        elif x in w :
          w[x].add(my_name)
        else :
          print('error')
          exit(9)

  for x in list(w.keys()) :
    if len(w[x]) > 1 :
      r.append(w[x])
  return r

def getGraphFromModel(model, **kwargs):
    """
    Create a `graphviz.Digraph` object that represents `model`.
    This function passes `kwargs` to `lbann.viz.getGraphFromPrototext`.
    """

    return getGraphFromPrototext(model.export_proto())

def getGraphFromPrototext(proto, format="pdf",
                          props=None, full=False, brief=False,
                          ranksep=0):
    """
    Create a `graphviz.Digraph` object from `proto`.
    The `format` argument is used as an extension when the resulting
    graph is rendered.
    """

    if props is None:
      props = Properties(
        os.path.join(
          os.path.dirname(os.path.abspath(__file__)),
          "properties",
          "properties.txt"))

    layers = [Layer(str(l).strip().split("\n")) for l in proto.layer]

    fixSequentialParents(layers)

    #get list of linked layer sets
    linked = getLinkedLayers(layers)

    #build a couple of maps
    edges = {}
    name_to_type = {}
    attributes = {}
    for layer in layers :
      name = layer.name()
      parents = layer.parents()

      #children = layer.children()
      attributes[name] = layer.attributes()
      type = layer.type()
      name_to_type[name] = type
      for p in parents :
        if p not in edges :
          edges[p] = set()
          edges[p].add(name)

    #write the dot file
    g = Digraph(format=format)
    g.attr("graph", ranksep=str(ranksep))

    for parent in edges.keys():
      type = name_to_type[parent]
      label = ''
      if brief:
        label = '<<font point-size="18">' + type + '</font>'
      else :
        label = '<<font point-size="18">' + type + '</font><br/>name: ' + parent
      if full :
        attr = attributes[parent]
        if len(attr) :
          label += '<br/>'
          for x in attr :
            label += x + '<br align="left"/>'

      label += ' >'

      g.node(
        parent,
        label=label,
        shape=props.shape(type),
        style="filled",
        fillcolor=props.color(type))

    #write edges
    for parent in list(edges.keys()) :
      type = name_to_type[parent]
      for child in edges[parent] :
        child_type = name_to_type[child]
        if type == 'slice' :
          g.edge(parent, child,
                 color="red", penwidth="2.0")
        elif type == 'split' :
          g.edge(parent, child,
                 color="darkorange", penwidth="2.0")
        elif child_type == 'sum' :
          g.edge(parent, child,
                 color="deepskyblue", penwidth="2.0")
        else :
          g.edge(parent, child)

    #alternatove to above: use subgraphs
    #write linked layer subgraphs
    for n, x in enumerate(linked):
      with g.subgraph(name="cluster_"+str(n), style="dashed") as sg:
        for node in x:
          sg.node(node)

    return g
