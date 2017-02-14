#!/usr/bin/python
from sys import *
from common import *
import os

usage = '''
usage: %s outputFilename [inputFilename]

if inputFilename is given, it should be an existing prototext
file, which you can then edit. Otherwise you'll be building
a model from a template.
''' % argv[0]

if len(argv) <2 :
  print usage
  exit(9)


#read in existing prototext file, or start from scratch.
mode = 'fresh_start'
if len(argv) == 3 :
  data = loadPrototextFile(argv[2])
  writePrototextFile(argv[1], data)
  mode == 'edit_existing'
else :
  #if starting from scratch, go to edit screen for TrainingParams,
  #then edit screen for optimizer
  #since these will be used as defaults for (some) values in other sections
  data = {}
  addTrainingParamsDefaults(data)
  addPerformanceParamsDefaults(data)
  addOptimizerDefaults(data)
  addModelDefaults(data)
  addDataReaderDefaults(data)

  editSection(data, argv[1], 'training_params', TrainingParamsMsg)
  editSection(data, argv[1], 'performance_params')
  editSection(data, argv[1], 'optimizer')
  editSection(data, argv[1], 'model')
  editDataReaders(data, argv[1])

#main loop
while True :
  clear()
  print msg1
  a = raw_input('  press a number, followed by <enter>: ')

  if a == 'v' :
    writePrototextFile(argv[1], data)
    os.system('less ' + argv[1])
  if a == 'q' :
    writePrototextFile(argv[1], data)
    break

  if a == '0' :
    if not data.has_key('training_params') : addTrainingParamsDefaults(data)
    editSection(data, argv[1], 'training_params')

  if a == '1' :
    if not data.has_key('performance_params') : addPerformanceParamsDefaults(data)
    editSection(data, argv[1], 'performance_params')

  if a == '2' :
    if not data.has_key('network_params') : addNetworkParamsDefaults(data)
    editSection(data, argv[1], 'network_params')

  if a == '3' :
    if not data.has_key('system_params') : addSystemParamsDefaults(data)
    editSection(data, argv[1], 'system_params')

  if a == '4' :
    if not data.has_key('data_reader') : addDataReaderDefaults(data)
    editDataReaders(data, argv[1])

  if a == '5' :
    if not data.has_key('model') : addModelDefaults(data)
    editSection(data, argv[1], 'model')

  if a == '6' :
    if not data.has_key('optimizer') : addOptimizerDefaults(data)
    editSection(data, argv[1], 'optimizer')

  if a == '7' : #layers
    if not data.has_key('layer') : addLayerDefaults(data)
    editLayers(data, argv[1])

  if a == '8' : #metric
    underConstruction()
  if a == '9' : #callback
    underConstruction()

  clear()

