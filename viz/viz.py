#!/usr/bin/python
from sys import *
import os
import pprint
from properties import *
from layer import *

usage = '''
usage: %s model_fn.prototext [output=<string>] [format=<string>] [prop=<string>] [full=1] [brief=1] [ranksep=<double>]

where: "output" is the output file basename; default is "graph"

       "format" refers to the output file; default is pdf, so the default
       output file is "graph.pdf" You can find a list of other formats at:
       http://www.graphviz.org/content/output-formats or just try your
       favorite (gif, png, jpg, etc) -- it's probably supported!
       Note: some formats may take a while to render, so be patient.

       "prop" is the name of the properties file; default is "properties.txt"
       The properties file is a simple text file that lists colors and
       shapes for the various layer types

       if "full=1" is present, all layer attributes are printed (e.g,
       num_neurons, has_bias, etc). The default is to print only the
       layer type and layer name

       if "brief=1", only the nodes' layer types are printed

       use "ranksep=<double> to increase of decrease the verticle distance
       between nodes. Hint: start with "ranksep=.75" and adjust up or down
       from there

note: the ordering of the optional params doesn't matter

note: in addition to the output file, an intermediate file called
      'graph.dot' will be written
''' % argv[0]


#=====================================================================
def parsePrototext(fn) :
  '''returns a list of Layers'''
  a = open(fn).readlines()
  r = []
  for j in range(len(a)) :
    if a[j].find('layer {') != -1 and a[j].find('#') == -1 :
      ell = Layer(a[j:])
      r.append(Layer(a[j:]))
  return r

#=====================================================================

if len(argv) < 2 :
  print usage
  exit(9)

#parse cmd line
output_fn = "graph"
output_format = "pdf"
prop_fn = "properties.txt"
full = False
brief = False
ranksep=0
for j in range(2, len(argv)) :
  t = argv[j].split('=')
  if t[0] == 'output' :
    output_fn = t[1]
  elif t[0] == 'format' :
    output_format = t[1]
  elif t[0] == 'prop' :
    prop_fn = t[1]
  elif t[0] == 'full' :
    full = True
  elif t[0] == 'brief' :
    brief = True
  elif t[0] == 'ranksep' :
    ranksep = float(t[1])
  else :
    print 'badly formed or unknown cmd line option:', argv[j]
    print '================================================================'
    print
    print usage
    exit(9)

#=====================================================================
def fixSequentialParents(layers) :
  '''a hack for models that don't contain parent and children fields'''
  num_layers_with_parents = 0
  num_layers_with_children = 0
  for layer in layers :
    if len(layer.parents()) != 0 : num_layers_with_parents += 1
    if len(layer.children()) != 0 : num_layers_with_children += 1
  if num_layers_with_parents == 0 :
    print
    print 'NOTE: this model does not appear to have any parent fields;'
    print '      dealing with that ...'
    print
    assert(num_layers_with_children == 0)
  for j in range(1, len(layers)) :
    layers[j].setParents(layers[j-1])


#=====================================================================
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
        if w.has_key(my_name) : 
          w[my_name].add(x)
        elif w.has_key(x) : 
          w[x].add(my_name)
        else :
          print 'error'
          exit(9)

  for x in w.keys() :
    if len(w[x]) > 1 :
      r.append(w[x])
  return r

#=====================================================================

#load properties database
props = properties(prop_fn)

#parse the prototext file; 'layers' is a list of Layer objects
layers = parsePrototext(argv[1])
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
  children = layer.children()
  attributes[name] = layer.attributes()
  type = layer.type()
  name_to_type[name] = type
  for p in parents :
    if not edges.has_key(p) :
      edges[p] = set()
    edges[p].add(name)
  if not edges.has_key(name) :
    edges[name] = set()
  for c in children :
    edges[name].add(c)

#write the dot file
out = open('graph.dot', 'w')
out.write('digraph xyz {\n')
if ranksep > 0 :
  out.write('graph[ranksep="' + str(ranksep) + '"]\n')

#write vertices
for parent in edges.keys() :
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
  label += '> '

  out.write('  ' + parent + '[label=' + label + ' shape=' + props.shape(type) + ', style=filled, fillcolor=' + props.color(type) + ']\n')

#write edges
for parent in edges.keys() :
  type = name_to_type[parent]
  for child in edges[parent] :
    child_type = name_to_type[child]
    if type == 'slice' : 
      out.write(parent + ' -> ' + child + '[color=red, penwidth=2.0];')
    elif type == 'split' : 
      out.write(parent + ' -> ' + child + '[color=darkorange, penwidth=2.0];')
    elif child_type == 'sum' : 
      out.write(parent + ' -> ' + child + '[color=deepskyblue, penwidth=2.0];')
    else :
      out.write(parent + ' -> ' + child + '[];\n')

#write linked layer edges
# commenting this out, since it makes the graph pretty useless and unreadable!
'''
for layer in layers :
  name = layer.name()
  links = layer.linkedLayers()
  for other in links :
    if other != name :
      pass
      print name, other
      out.write(name + ' -> ' + other + '[color=red, dir=none]\n')
'''

#alternatove to above: use subgraphs
#write linked layer subgraphs    
n = 0
for x in linked :
  out.write('subgraph cluster_' + str(n) + ' {\n')
  out.write('   style=dashed;\n')
  n += 1
  for node in x :
    out.write('   '+ node + ';\n')
  out.write('}\n')

out.write('}\n')
out.close()

#run graphviz
cmd = 'dot -T' + output_format + ' graph.dot -o' + output_fn + '.' + output_format
print 
print 'about to run:', cmd
os.system(cmd)
