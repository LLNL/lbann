#import curses
import pprint
#from collections import OrderedDict
from defaults import *

def initModel() :
  pass  

def clear() :
  print '\n\n\n\n\n\n\n\n\n\n'
  print '\n\n\n\n\n\n\n\n\n\n'
  print '\n\n\n\n\n\n\n\n\n\n'
  print '\n\n\n\n\n\n\n\n\n\n'
  print '\n\n\n\n\n\n\n\n\n\n'
  print '\n\n\n\n\n\n\n\n\n\n'

msg1 = '''



  You can add and edit the following:

    0. Training Params
    1. Performance Params
    2. Network Params
    3. System Params
    4. DataReader
    5. Model
    6. Optimizer
    7. Layers
    8. Metric
    9. Callback

    v. View current prototext file (using less)
    q. Quit

'''

#=============================================================================
def findMatchingBrace(data, first) :
  n = first
  num_nested = 0
  while n < len(data) :
    if data[n].find('}') != -1 and num_nested == 0 :
      return n
    else :
      num_nested -= 1
    if data[n].find('{') :
      num_nested += 1
    n += 1
  return None

#=============================================================================
def findNested(data, k) :
  name = data[k].split()
  name = name[0]
  k += 1
  nested = []
  my_data = []
  num_nested = 0
  end = 0
  n_start = 0 
  while k < len(data) :
    if data[k].find('}') != -1 and num_nested == 0 :
      end = k
      break
    if data[k].find('}') == -1 and data[k].find('{') == -1 and num_nested == 0 :
      t = data[k].strip()
      x = t.find(':')
      assert(x != -1)
      key = t[:x]
      val = t[x+2:]
      val = val.strip('"')
      my_data.append( [key, val] )
    elif num_nested == 1 and  data[k].find('}') != -1 :
        nested.append( (n_start, k) )
        num_nested -= 1
    elif num_nested > 1 and  data[k].find('}') != -1 :
        num_nested -= 1
    if data[k].find('{') != -1 :
      if num_nested == 0 :
        n_start = k
      num_nested += 1

    k += 1  
  return(name, my_data, nested, end)
    
    
#=============================================================================
def printme(name, my_data, nested, end, msg = None) :
  return
  print '=================================================='
  if msg != None : print msg
  print 'name:', name
  print 'my_data:',
  pprint.pprint(my_data)
  print 'nested:'
  pprint.pprint(nested)

#=============================================================================
#yeah, could be much neater if I used recurssion ...
def loadPrototextFile(fn) :
  data = {}

  f = open(fn).readlines()
  j = 0
  while True :
    if f[j].find('{') != -1 and f[j][0] != ' ' :
      [name0, my_data0, nested0, end0] = findNested(f, j)
      j = end0
      if not data.has_key(name0) : data[name0] = []
      data[name0].append( [name0, ''] )
      if len(my_data0) :
        data[name0].append( my_data0 )


    #1st possibly nested layer
    if len(nested0) :
      for x in nested0 :
        [name1, my_data1, nested1, end1] = findNested(f, x[0])
        if name0 == 'data_reader' :
          data[name0].append( [name1, ''] )
          if len(my_data1) :
            data[name0].append(my_data1)
        elif name0 == 'model' and name1 == 'layer' :
            if not data.has_key('layer') :
              data['layer'] = []
        elif name0 == 'data_reader' and name1 == 'optimizer' :
          assert(not data.has_key('optimizer'))
          data[name1].append( [[name1, ''], my_data1] )
        else : #only thing left is params    
            data[name1] = [ [name1, ''], my_data1 ]

        
        #2nd possibly nested layer
        if len(nested1) :
          for y in nested1 :
            [name2, my_data2, nested2, end2] = findNested(f, y[0])
            
            assert(name0 == 'model')
            assert(name1 == 'layer')
            assert(data.has_key('layer'))
            data['layer'].append( [name2, ''] )
            if len(my_data2) :
              data['layer'].append( my_data2 )

            #3rd possibly nested layer
            if len(nested2) :
              for z in nested2 :
                [name3, my_data3, nested3, end3] = findNested(f, z[0])
                if len(nested) :
                  print 'ERROR: nesting more than three deep; this program needs revision!'
                  print 'please contact dave hysom'
                  exit(9)
       

    j += 1
    if j >= len(f) : break
  return data  

#=============================================================================
#Note: any field in lbann.proto that has type 'string' should appear in
#      the following set. This is so the user doesn't need to worry about
#      which entries are strings during edits.
Parens = set(
    ['activation_type', 'dataset_root_dir', 'save_image_dir', 'parameter_dir',
     'train_file', 'test_file', 'summary_dir', 'dump_dir',
     'role', 'file_dir', 'image_dir', 'label_file', 'name', 'objective_function',
     'metric', 'callback', 'activation_type', 'weight_initialization', 
     'optimizer', 'regularizer', 'pool_mode', 'host_name',
     'network_str'
    ]
  )  

def parens( x ) :
  key = x[0]
  value = x[1]
  if key in Parens :
    return '"' + value + '"'
  return value

#=============================================================================
def getDataReaders(data) :
  r = []
  if len(data['data_reader']) > 1 :
    w = data['data_reader'][1:]
    if len(w) > 1 :
      for j in range(0, len(w), 2) :
        cur_reader = []
        cur_reader.append(w[j])
        try :
          for n in w[j+1] :
            cur_reader.append(n)
        except:   
          print "error in getDataReaders; data['data_reader']", 
          pprint.pprint(data['data_reader'])
          print "data['data_reader'][1:]:", 
          pprint.pprint(data['data_reader'][1:])
          exit(9)
        r.append(cur_reader)  
  return r  

#=============================================================================
def writePrototextFile(fn, data) :
  o = open(fn, 'w')

  if data.has_key('data_reader') :
    readers = getDataReaders(data)
    o.write('data_reader {\n')
    for reader in readers :
      o.write('  ' + reader[0][0]+ ' {\n')
      for j in range(1, len(reader)) :
        o.write('    ' + reader[j][0] + ': ' + parens(reader[j]) + '\n')
      o.write('  }\n')
    o.write('}\n')

  if data.has_key('model') :
    o.write('model {\n')
    for d in data['model'][1] :
        o.write('  ' + d[0] + ': ' + parens(d) + '\n')

  if data.has_key('layer') :
    for i in range(len(data['layer'])) :
      if data['layer'][i][1] == '' :
        o.write('  layer {\n')
        o.write('    ' + data['layer'][i][0] + ' {\n')
      else :
        for d in data['layer'][i] :
          o.write('      ' + d[0] + ': ' + parens(d) + '\n')

      try :
       if i == len(data['layer'])-1 :
        o.write('    }\n')  
        o.write('  }\n')  
       elif data['layer'][i+1][1] == '' :
        o.write('    }\n')  
        o.write('  }\n')  
      except :
         print 'writePrototextFile ERROR:', 
         pprint.pprint(data['layer'])

  if data.has_key('optimizer') :
    o.write('  optimizer {\n')
    for x in data['optimizer'][1] :
      o.write('    ' + x[0] + ': ' + parens(x) + '\n')
    o.write('  }\n')  
  o.write('}\n')  

  if data.has_key('performance_params') :
    o.write('performance_params {\n')
    for x in data['performance_params'][1] :
        o.write('  ' + x[0] + ': ' + parens(x) + '\n')
    o.write('}\n')

  if data.has_key('system_params') :
    o.write('system_params {\n')
    for x in data['system_params'][1] :
        o.write('  ' + x[0] + ': ' + parens(x) + '\n')
    o.write('}\n')

  if data.has_key('training_params') :
    o.write('training_params {\n')
    z = 0
    for x in data['training_params'][1] :
        o.write('  ' + x[0] + ': ' + parens(x) + '\n')
    o.write('}\n')

  if data.has_key('network_params') :
    o.write('network_params {\n')
    for x in data['network_params'][1] :
        o.write('  ' + x[0] + ' ' + parens(x) + '\n')
    o.write('}\n')

  o.close()

#=============================================================================
def showSection(data, name, msg) :
  clear()
  print 
  if msg : print msg
  print
  print '  You are in edit mode for section:', name
  print 
  idx = 0
  for t in data[name][1] :
    print '    ' + str(idx) + ' - ' + t[0] + ': ' + t[1]
    idx += 1
  print 
  print '    s - save and return'
  print '    d - delete this section (experimental! may have unexpected consequences!)'
  print

#=============================================================================
def editSection(data, fn, section, msg = None) :
  while True :
    showSection(data, section, msg)

    a = raw_input('press <number><enter> to edit a field: ')
    try :
      if not (a == 's' or a == 'd') :
        a = int(a)
        assert(a >= 0 and a <= len(data[section][1]))
    except :  
      clear()
      raw_input('bad entry; please press <enter> to try again')
      continue
    print
    if a == 's': return
    if a == 'd':
      del(data[section]) 
      return
    clear()
    b = raw_input('enter new value for: ' + data[section][1][a][0] + ': ')
    data[section][1][a][1] = b
    writePrototextFile(fn, data)
  return data

#=============================================================================
def underConstruction() :
  clear()
  for j in range(10) : 
    print
  print '          >>>>> Under Construction <<<<<'
  print
  print
  print
  print
  raw_input('press <enter> to return')

#=============================================================================
LayerNames = ['input_distributed_minibatch_parallel_io',
            'convolutional',
            'fully_connected',
            'pooling',
            'softmax',
            'target_distributed_minibatch_parallel_io']

def getLayerChoice(data) :
  global LayerNames
  clear()
  print
  print '  You can add any of the following Layers:'
  print 
  for j in range(len(LayerNames)) :
    print '    ' + str(j) + ' - ' + LayerNames[j]
  print
  print '    s - save and return'
  print
  a = raw_input('Enter number for Layer to be added: ')
  try :
    a = int(a)
  except :
    return 'none'
  return LayerNames[a]

def addInputMPIO(data, fn) :
  data['input_distributed_minibatch_parallel_io'] = []
  data['input_distributed_minibatch_parallel_io'].append(['input_distributed_minibatch_parallel_io', ''])
  data['input_distributed_minibatch_parallel_io'].append([
    ['num_parallel_readers', '0']
    #TODO regularizers
  ])
  editSection(data, fn, 'input_distributed_minibatch_parallel_io')
  tmp = data['input_distributed_minibatch_parallel_io']
  del(data['input_distributed_minibatch_parallel_io'])
  data['layer'].append(tmp[0])
  data['layer'].append(tmp[1])

def addConvolutional(data,fn) :
  data['convolutional'] = []
  data['convolutional'].append(['convolutional', ''])
  data['convolutional'].append([
    ['num_input_channels', '0'],
    ['input_dims', '2 2 2'],
    ['num_output_channels', '0'],
    ['filter_dims', '2 2 2'],
    ['conv_pads', '2 2 2'],
    ['conv_strides', '2 2 2'],
    ['activation_type', getTrainingParamValue(data, 'activation_type')],
    ['weight_initialization', getTrainingParamValue(data, 'weight_initialization')]
    #TODO regularizers
  ])
  editSection(data, fn, 'convolutional')
  tmp = data['convolutional']
  del(data['convolutional'])
  data['layer'].append(tmp[0])
  data['layer'].append(tmp[1])

def addFullyConnected(data, fn) :
  data['fully_connected'] = []
  data['fully_connected'].append(['fully_connected', ''])
  data['fully_connected'].append([
    ['num_neurons', '0'],
    ['activation_type', getTrainingParamValue(data, 'activation_type')],
    ['weight_initialization', getTrainingParamValue(data, 'weight_initialization')],
    ['optimizer', 'TODO']
    #TODO regularizers
    #['weight_initialization', getTrainingParamValue(data, 'weight_initialization')]
  ])
  editSection(data, fn, 'fully_connected')
  tmp = data['fully_connected']
  del(data['fully_connected'])
  data['layer'].append(tmp[0])
  data['layer'].append(tmp[1])

def addPooling(data,fn) :
  data['pooling'] = []
  data['pooling'].append(['pooling', ''])
  data['pooling'].append([
    ['num_dims', '0'],
    ['num_channels', '0'],
    ['input_dims', '2 2 2'],
    ['pool_dims', '2 2 2'],
    ['pool_pads', '2 2 2'],
    ['pool_strides', '2 2 2'],
    ['pool_mode', 'max'], #TODO: add default to TrainingParams????
    ['activation_type', getTrainingParamValue(data, 'activation_type')]
    #TODO regularizers
  ])
  editSection(data, fn, 'pooling')
  tmp = data['pooling']
  del(data['pooling'])
  data['layer'].append(tmp[0])
  data['layer'].append(tmp[1])

def addSoftMax(data,fn) :
  data['softmax'] = []
  data['softmax'].append(['softmax', ''])
  data['softmax'].append([
    ['num_neurons', '0'],
    ['weight_initialization', getTrainingParamValue(data, 'weight_initialization')]
  ])
  editSection(data, fn, 'softmax')
  tmp = data['softmax']
  del(data['softmax'])
  data['layer'].append(tmp[0])
  data['layer'].append(tmp[1])

def addTargetMPIO(data, fn) :
  data['target_distributed_minibatch_parallel_io'] = []
  data['target_distributed_minibatch_parallel_io'].append(['target_distributed_minibatch_parallel_io', ''])
  data['target_distributed_minibatch_parallel_io'].append([
    ['num_parallel_readers', '0'],
    ['shared_data_reader', 'true']
    ['for_regression', 'false']
    #TODO regularizers
  ])
  editSection(data, fn, 'target_distributed_minibatch_parallel_io')
  tmp = data['target_distributed_minibatch_parallel_io']
  del(data['target_distributed_minibatch_parallel_io'])
  data['layer'].append(tmp[0])
  data['layer'].append(tmp[1])


def editLayers(data, fn) :
  s = getLayerChoice(data)
  if s == 'none' : return
  if s == 'input_distributed_minibatch_parallel_io' : addInputMPIO(data, fn)
  elif s == 'convolutional' : addConvolutional(data,fn)
  elif s == 'fully_connected' : addFullyConnected(data, fn)
  elif s == 'pooling' : addPooling(data,fn)
  elif s == 'softmax' : addSoftMax(data,fn)
  elif s == 'target_distributed_minibatch_parallel_io' : addTargetMPIO(data, fn)

#=============================================================================
def printDataReaderScreen(data) :
  clear()
  print
  print
  readers = getDataReaders(data)
  print 'There are currently', len(readers), 'DataReader(s)'
  print 

  idx = 0
  for d in readers :
    print '    ' + str(idx) + ' - ' + d[0][0] + ': ' + d[1][1]
    idx += 1
  print 
  print '    Enter number to edit an existing data reader, or:'
  print '    a - add a new reader'
  print '    d - delete all DataReaders'
  print '    s - save and return'
  print

DataReaderNames = ['cifar10', 'imagenet', 'mnist', 'nci', 'nci_regression']
def getDataReaderChoice(data) :
  clear()
  print
  print '  Available DataReaders:'
  print 
  for j in range(len(DataReaderNames)) :
    print '    ' + str(j) + ' - ' + DataReaderNames[j]
  print
  a = raw_input('Enter number for DataReader to be added: ')
  a = int(a)
  return DataReaderNames[a]

def addMnist(data, fn) :
  global BatchSize
  data['mnist'] = []
  data['mnist'].append( ['mnist', ''] )
  data['mnist'].append( 
    [ ['role', 'train'], 
      ['batch_size', getTrainingParamValue(data, 'mb_size')],
      ['shuffle', getTrainingParamValue(data, 'shuffle_training_data')],
      ['file_dir', 'none'],
      ['image_file', 'none'],
      ['label_file', 'none'],
      ['percent_samples', getTrainingParamValue(data, 'percentage_training_samples')]
    ])  
  editSection(data, fn, 'mnist')
  tmp = data['mnist'] 
  del(data['mnist'])
  data['data_reader'].append(tmp[0])
  data['data_reader'].append(tmp[1])
  setTrainingParamValue(data, 'mb_size', tmp[1][1][1])

def addDataReader(data, fn) :
  s = getDataReaderChoice(data)
  if s == 'mnist' : addMnist(data, fn)
  elif s == 'imagenet' : underConstruction()
  elif s == 'nci' : underConstruction()
  elif s == 'nci_regression' : underConstruction()
  elif s == 'cifar10' : underConstruction()

def editDataReaders(data, fn) :
  while True :
    readers = getDataReaders(data)
    printDataReaderScreen(data)
    a = raw_input('what would you like to do? ')
    try :
      if not (a == 'a' or a == 'd' or a == 's') :
        a = int(a)
        assert(a >= 0 and a < len(readers))
    except :  
      clear()
      raw_input('bad entry; please press <enter> to try again')
      continue
    print
    if a == 'a':
      addDataReader(data, fn)
      writePrototextFile(fn, data)
    elif a == 'd':
      del data['data_reader']
      return
    elif a == 's' :
      return data
    else :
      pass
      #TODO
      #editDataReader(data, readers[a])
  return data

#=============================================================================
TrainingParamsMsg = '''
TrainingParams contain global defaults. Some of these can be
overridden, e.g, in the Layer sections. Once you have constructed
a Layer, changing a TrainingParam will not affect its settings.
'''
#=============================================================================
