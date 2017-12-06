#!/usr/bin/env python
import commands
import re
import sys
import os
import copy
import pprint

usage = '''
usage: %s input_filename.prototext output_filename.prototext
function: expands motifs
''' % sys.argv[0]

#============================================================================
def runme(cmd, msg='') :
  result = commands.getstatusoutput(cmd)
  if result[0] != 0 :
    sys.stderr.write('this shell command failed: ' + cmd + '\n')
    exit(9)
  return(result[1])

#============================================================================
def build_descriptor_pb2(p_path) :
  runme('cd  ' + p_path + '; python setup.py build')
  
#============================================================================
def get_models_dir() :
  return runme('git rev-parse --show-toplevel') + '/model_zoo/models'

#============================================================================
def set_python_search_path() :
  lbann_dir = runme('git rev-parse --show-toplevel')
  host = commands.getstatusoutput("hostname")
  host = re.sub("\d+", "", host[1])
  p_path = lbann_dir + '/build/' + host + '.llnl.gov/download/protobuf/source/python'
  sys.path.append(p_path)
  global lbann_pb2
  import lbann_pb2
  global txtf
  import google.protobuf.text_format as txtf
  return p_path

#============================================================================
def has_motifs(model) :
  for layer in model.layer :
    if layer.HasField("motif_layer"):
      return True
  return False

#============================================================================
def load_prototext(fn) :
  a = lbann_pb2.LbannPB()
  f = open(fn).read()
  print '\n in load_prototext\n'
  pb = txtf.Merge(f, a)
  return pb

#============================================================================
def compile_lbann_proto() :
  lbann_dir = runme('git rev-parse --show-toplevel')
  proto_dir = lbann_dir + '/src/proto'
  proto_fn = lbann_dir + '/src/proto/lbann.proto'
  host = runme("hostname")
  host = re.sub("\d+", "", host)
  protoc = lbann_dir + '/build/' + host + '.llnl.gov/bin/protoc'
  r = commands.getstatusoutput(protoc + ' -I=' + proto_dir + ' --python_out=. ' + proto_fn)
  
#============================================================================
def write_output(p, fn) :
  out = open(fn, 'w')
  s = txtf.MessageToString(p)
  out.write(s)
  out.close()

#============================================================================
def getMotifInputOrOutputLayerName(layer, motifs, which) :
  assert(which == 'parent' or which == 'child')
  retval = []
  id = layer.motif_layer.motif_id
  assert(motifs.has_key(id))
  motif = motifs[id]
  a = motif.name
  a = a.replace('motif', '')
  #base_name = layer.name + '_' + a
  base_name = layer.name
  for m_layer in motif.layer :
    if which == 'parent' :
      if m_layer.parents == '' :
        name = base_name + '_' + m_layer.name
        retval.append(name)
    else :
      if m_layer.children == '' :
        name = base_name + '_' + m_layer.name
        retval.append(name)
  
  assert(len(retval) == 1)
  return retval

#============================================================================
def expandMotifLayer(motif_layer, motif) :
      expanded_layers = []

      a = motif.name
      a = a.replace('motif', '')
      base_name = motif_layer.name

      old_to_new_name = {}

      #hash the variables
      variables = {}
      skip_me = {}
      for v in motif_layer.motif_layer.variable :
        t = v.split()
        name = t[0]
        if name == 'do_not_use' :
          skip_me[t[1]] = None
        else :  
          if not variables.has_key(name) :
            variables[name] = set()
          variables[t[0]].add(v)

      for layer in motif.layer :
        if skip_me.has_key(layer.name) :
          skip_me[layer.name] = layer

      for x in skip_me.keys() :
        assert(skip_me[x] != None)
  
      #loop over the layers in the motif; turn each one into an actual layer
      #in the model
      for layer in motif.layer :
        if skip_me.has_key(layer.name) :
          print 'not using optional layer:', layer.name
        else :
          new_layer = lbann_pb2.Layer()
          new_layer.CopyFrom(layer)
          print 'constructing layer:', layer.name
  
          #get the variables for this layer, if any
          fake_name = new_layer.name
          vv = None
          if variables.has_key(fake_name) :
            vv = variables[fake_name]
            print '  layer has these variables:', vv
  
          #set a unique name for the layer
          org_name = new_layer.name
          name = base_name + '_' + org_name
          old_to_new_name[org_name] = name
          new_layer.name = name
          if new_layer.parents == "" :
            new_layer.parents = motif_layer.parents
          if new_layer.children == "" :
            new_layer.children = motif_layer.children
  
          #deal with layers that have variables
          if not vv :
            expanded_layers.append(new_layer)
          else :
            string_layer = txtf.MessageToString(new_layer)
            list_layer = string_layer.split('\n')
            for j in range(len(list_layer)) :
              for tuple in vv :
                t = tuple.split()
                field = t[1]
                val = t[2]
                if list_layer[j].find(field) != -1 :
                  h = list_layer[j].rfind(':')
                  assert(h != -1)
                  list_layer[j] = list_layer[j][:h+1] + ' ' + val
                  #assert(list_layer[j].find('-1') != -1)
                  #list_layer[j] = list_layer[j].replace('-1', val)
            string_layer = '\n'.join(list_layer)
            tmp = lbann_pb2.LbannPB()
            tmp = lbann_pb2.Layer()
            txtf.Merge(string_layer, tmp)
            expanded_layers.append(tmp)
      
      #fix parent and child names
      for layer in expanded_layers :
        t = layer.parents.split()
        parents = []
        for p in t :
          if skip_me.has_key(p) :
            n = p
            while skip_me.has_key(n) :
              pp = skip_me[n].parents.split()
              assert(len(pp) == 1)
              n = pp[0]
            assert(old_to_new_name.has_key(n))
            parents.append(old_to_new_name[n])
          elif old_to_new_name.has_key(p) :
            parents.append(old_to_new_name[p])
          else :
            parents.append(p)
        pp = ' '.join(parents)
        layer.parents = pp

        t = layer.children.split()
        children = []
        for p in t :
          if skip_me.has_key(p) :
            n = p
            while skip_me.has_key(n) :
              pp = skip_me[n].children.split()
              assert(len(pp) == 1)
              n = pp[0]
            assert(old_to_new_name.has_key(n))
            parents.append(old_to_new_name[n])
          if old_to_new_name.has_key(p) :
            children.append(old_to_new_name[p])
          else :
            pass
            #print 'xxxxx layer:', layer.name, ' old_to_new_name not found for child:', p
        pp = ' '.join(children)
        layer.children = pp
      return expanded_layers

#============================================================================
def fixNames(motifs, model) :
  #build maps: motif_layer.name -> name of 1st layer in the expansion
  #            motif_layer.name -> name of last layer in the expansion
  input_names = {}
  output_names = {}
  motif_layer_names = set()
  for layer in model.layer :
    if layer.HasField("motif_layer") :
      motif_layer_names.add(layer.name)
      r = getMotifInputOrOutputLayerName(layer, motifs, 'parent')
      input_names[layer.name] = r
      r = getMotifInputOrOutputLayerName(layer, motifs, 'child')
      output_names[layer.name] = r

  #pprint.pprint(input_names)
  #pprint.pprint(output_names)
  for layer in model.layer :
      parents = layer.parents
      t = parents.split()
      p = ''
      for name in t :
        if name in motif_layer_names :
          assert(name in output_names.keys())
          for pp in output_names[name] :
            p += pp
        else :
          p += name + ' '
      layer.parents = p.strip()

      children = layer.children
      t = children.split()
      p = ''
      for name in t :
        if name in motif_layer_names :
          assert(name in input_names.keys())
          for pp in input_names[name] :
            p += pp
        else :
          p += name + ' '
      layer.children = p.strip()

#============================================================================
def main(argv) :
  global usage
  if len(argv) != 3 :
    print usage
    exit(9)

  compile_lbann_proto()
  p_path = set_python_search_path()
  build_descriptor_pb2(p_path)
  
  pb = load_prototext(argv[1])
  model = pb.model
  if not has_motifs(model) :
    print 'The input prototext file does not contain motifs; the output file'
    print '"' + argv[2] + '" will contain identical information as the input file: "' + argv[1] + '"'
    write_output(pb, argv[2])
    exit(0)
  
  #build table: motif name -> motif
  motif_defs = pb.motif_definitions
  motifs = {}
  for m in motif_defs.motif :
    motifs[m.name] = m
  
  #make copy of prototext, then delete the layers in the copy
  b = lbann_pb2.LbannPB()
  b.CopyFrom(pb)
  del b.model.layer[:]

  #error check
  model_name = model.name
  known_models = set(['dag_model', 'sequential_model'])
  print 'model_name:', model_name
  if model_name not in known_models :
    print 'nothing known about the model named:', model_name
    print 'please update this code, or fix the prototext file'
    print 'known models:',
    for m in known_models : 
      print m
    exit(9)

  is_sequential = False
  if model_name == 'sequential_model' :
    is_sequential = True

  #fix the names for layers that are not motif_layers, but whose
  #parents and/or children are motif_layers
  fixNames(motifs, model)

  #loop over the layers in the input prototext; expand the motif layers
  for layer in model.layer :
    if not layer.HasField("motif_layer") :
      b.model.layer.extend([layer])
    else :
      #get the requested motif
      id = layer.motif_layer.motif_id
      assert(motifs.has_key(id))
      motif = motifs[id]
      expanded_motif_layers = expandMotifLayer(layer, motif)
      for x in expanded_motif_layers :
        b.model.layer.extend([x])

  b.motif_definitions.Clear()
  print 'calling write_output'
  write_output(b, argv[2])
  
if __name__ == "__main__" :
  main(sys.argv)
