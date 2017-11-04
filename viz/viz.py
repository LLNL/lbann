#!/usr/bin/python
from sys import *
import os
import pprint
from properties import *
from layer import *

usage = '''
usage: %s model_fn.prototext [output=<string>] [format=<string>] [prop=<string>] [full=1]

where: "output" is the output file basename; default is "graph"

       "format" refers to the output file; default is pdf, so the default
       output file is "graph.pdf" You can find a list of other formats at:
       http://www.graphviz.org/content/output-formats or just try your
       favorite (gif, png, jpg, etc) -- it's probably supported!

       "prop" is the name of the properties file; default is "properties.txt"
       The properties file is a simple text file that lists colors and
       shapes for the various layer types

       if "full=1" is present, all layer attributes are printed (e.g,
       num_neurons, has_bias, etc). The default is to print only the
       layer type and layer name

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
  else :
    print 'badly formed or unknown cmd line option:', argv[j]
    print '================================================================'
    print
    print usage
    exit(9)


#load properties database
props = properties(prop_fn)

#parse the prototext file; 'layers' is a list of Layer objects
layers = parsePrototext(argv[1])

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

#write vertices
for parent in edges.keys() :
  type = name_to_type[parent]
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
  for child in edges[parent] :
    out.write(parent + ' -> ' + child + ';\n')

#write linked layer edges
# commenting this out, since it makes the graph pretty useless and unreadable!
for layer in layers :
  name = layer.name()
  links = layer.linkedLayers()
  for other in links :
    if other != name :
      pass
      #print name, other
      #out.write(name + ' -> ' + other + '[color=red, dir=none]\n')
    

out.write('}\n')
out.close()

#run graphviz
cmd = 'dot -T' + output_format + ' graph.dot -o' + output_fn + '.' + output_format
os.system(cmd)
