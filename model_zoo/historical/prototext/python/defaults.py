import pprint
#=============================================================================
def addSystemParamsDefaults(data) :
  data['system_params'] = []
  data['system_params'].append( ['system_params', ''] )
  t = [
    ['host_name', ''],
    ['num_nodes', '-1'],
    ['num_cores', '-1'],
    ['tasks_per_node', '-1']
  ]  
  data['system_params'].append(t)
  pass

#=============================================================================
def addNetworkParamsDefaults(data) :
  data['network_params'] = []
  data['network_params'].append( ['network_params', ''] )
  t = [
    ['network_str', '']
  ]  
  data['network_params'].append(t)

#=============================================================================
def addTrainingParamsDefaults(data) :
  data['training_params'] = []
  data['training_params'].append( ['training_params', ''] )

  #attempting to put the params most likely needed to be modified at top of llist
  #most (not all) of the defaults are from lbann/src/lbann_params.cpp
  t = [
    ['epoch_count' , '10'],
    ['mb_size' , '192'],
    ['learn_rate' , '0.3'],
    ['learn_rate_method' , 'RMSprop', '(Adagrad, RMSprop, Adam, SGD)'],
    ['lr_decay_rate' , '0.5'],
    ['lr_decay_cycles' , '5000'],
    ['lr_momentum' , '0.9'],
    ['activation_type' , 'sigmoid'],
    ['weight_initialization' , 'glorot_uniform'], #no default in lbann_params!
    ['dropout' , '-1.0'],

    ['shuffle_training_data' , 'true'],
    ['enable_profiling' , 'false'],
    ['random_seed' , '42'],
    ['percentage_training_samples', '0.8'],
    ['percentage_validation_samples', '0.2'],
    ['percentage_testing_samples', '1.0'],
    ['test_with_train_data', '0'],
    ['epoch_start' , '0'],
    ['lambda' , '0.0'],
    ['dataset_root_dir', 'none'],
    ['save_image_dir', 'none'],
    ['parameter_dir', 'none'],
    ['save_model', 'false'],
    ['load_model', 'false'],
    ['ckpt_epochs', '0'],
    ['ckpt_steps', '0'],
    ['ckpt_secs', '0'],
    ['train_file', 'nont'],
    ['test_file', 'none'],
    ['summary_dir', 'none'],
    ['dump_weights', 'false'],
    ['dump_activations', 'false'],
    ['dump_gradients', 'false'],
    ['dump_dir', 'none'],
    ['intermodel_comm_method', '0'],
    ['procs_per_model', '0']
  ]
  data['training_params'].append(t)

#=============================================================================
def addPerformanceParamsDefaults(data) :
  if not data.has_key('performance_params') :
    data['performance_params'] = []
    t = [
      ['block_size', '1'],
      ['max_par_io_size', 0]
    ]  
    data['performance_params'].append(t)

#=============================================================================
def addModelDefaults(data) :
  if not data.has_key('model') :
    data['model'] = []
    data['model'].append( ['model', ''] )
    t = [ ['name', 'dnn'],
          ['objective_function', '"categorical_cross_entropy"'],
          ['num_epochs', getTrainingParamValue(data, 'epoch_count')],
          ['metric', '"categorical_accuracy"']
          #['mini_batch_size', getTrainingParamValue(data, 'mini_batch_size')]
        ]  
    data['model'].append(t)
    '''
    data['model'].append( ('name', 'dnn') )
    data['model'].append( ('objective_function', '"categorical_cross_entropy"') )
    data['model'].append( ('num_epochs', '10') )
    data['model'].append( ('metric', '"categorical_accuracy"') )
    data['model'].append( ('mini_batch_size', '192') )
    '''
  return data

#=============================================================================
def addLayerDefaults(data) :
  data['layer'] = []
  data['layer'].append( ['layer', ''] )

#=============================================================================
def addOptimizerDefaults(data) :
  data['optimizer'] = []
  data['optimizer'].append( ['optimizer', ''] )
  assert(data.has_key('training_params'))
  p = data['training_params']
  optimizer_learn_method = getTrainingParamValue(data, 'learn_rate_method')
  t = [
    ['name', getTrainingParamValue(data, 'learn_rate_method')],
    ['learn_rate', getTrainingParamValue(data, 'learn_rate')],
    ['momentum', getTrainingParamValue(data, 'lr_momentum')],
    ['decay', getTrainingParamValue(data, 'lr_decay_rate')],
    ['nesterov', 'false']
  ]  
  data['optimizer'].append(t)

#=============================================================================
def addDataReaderDefaults(data) :
  data['data_reader'] = []
  data['data_reader'].append( ['data_reader', ''] )

#=============================================================================
def getTrainingParamValue(data, s) :
  p = data['training_params'][1]
  for t in p :
    if t[0] == s :
      return t[1]
  return 'unknown'      

def setTrainingParamValue(data, s, value) :
  p = data['training_params'][1]
  success = False
  for t in p :
    if t[0] == s :
      success = True
  assert(success)
