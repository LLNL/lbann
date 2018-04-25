# Generate DenseNet prototext
# See https://github.com/pytorch/vision/blob/6f7e26be1018344a4d1015370dcedba7b772bbc1/torchvision/models/densenet.py
# See "Densely Connected Convolutional Networks" by Huang et. al

### DenseNet Specific #########################################################

def dense_net(prototext_file, version):
  f = open(prototext_file, 'w')
  f.write('model {\n')
  tab = ' '*2
  format_tuple = (tab,)*11
  f.write(
"""%sname: "directed_acyclic_graph_model"
%sdata_layout: "data_parallel"
%smini_batch_size: 256
%sblock_size: 256
%snum_epochs: 10
%snum_parallel_readers: 0
%sprocs_per_model: 0
%snum_gpus: -1
%s###################################################
%s# Objective function
%s###################################################
""" % ((tab,)*11))
  objective_function(f, tab)
  f.write(
"""%s###################################################
%s# Metrics
%s###################################################
""" % ((tab,)*3))
  metrics(f, tab)
  f.write(
"""%s###################################################
%s# Callbacks
%s###################################################
""" % ((tab,)*3))
  callbacks(f, tab)
  f.write(
"""%s###################################################
%s# Layers
%s###################################################
""" % ((tab,)*3))
  # batch_norm_size=4 specified in pytorch implementation
  batch_norm_size = 4
  # growth_rate, num_layers_tuple specified in "Densely Connected Convolutional Networks"
  # number_initial_features specified in pytorch implementation
  if version == 121:
    growth_rate = 32
    num_layers_tuple = (6, 12, 24, 16)
    number_initial_features = 64
  elif version == 161:
    growth_rate = 48
    num_layers_tuple = (6, 12, 36, 24)
    number_initial_features = 96
  else:
    raise Exception('Invalid version %d' % version)
  parent = initial_layer(f, tab, number_initial_features)
  num_blocks = 4
  for num in range(1, num_blocks + 1):
    num_layers = num_layers_tuple[num - 1]
    parent = dense_block(f, tab, num, num_layers, parent, batch_norm_size, growth_rate)
    if num != num_blocks:
      transition_num_features = number_initial_features + (num_layers * growth_rate)
      parent = transition_layer(f, tab, num, parent, transition_num_features / 2.0)
  classification_layer(f, tab, parent)
  f.write('}\n')

def initial_layer(f, tab, num_initial_channels):
  name = 'initial_layer'
  input_name = input_layer(f, tab, name, [], io_buffer='partitioned')
  # conv_dims_i=7, conv_strides_i=2 specified in "Densely Connected Convolutional Networks"
  # conv_pads_i=3 specified in pytorch implementation
  conv_block_name = conv_block(f, tab, name, [input_name], num_dims=2, num_output_channels=num_initial_channels, conv_dims_i=7, conv_pads_i=3, conv_strides_i=2)
  # pool_dims_i=3, pool_strides_i=2, pool_mode='max' specified in "Densely Connected Convolutional Networks"
  pool_name = pool(f, tab, name, [conv_block_name], num_dims=2, pool_dims_i=3, pool_strides_i=2, pool_mode='max')
  return pool_name

def conv_block(f, tab, name, parents, num_dims=None, num_output_channels=None, conv_dims_i=None, conv_pads_i=None, conv_strides_i=None):
  norm_name = norm(f, tab, name, parents, decay=0.9, scale_init=1.0, bias_init=0.0, epsilon=1e-5)
  relu_name = relu(f, tab, name, [norm_name])
  convolution_name = convolution(f, tab, name, [relu_name], num_dims=num_dims, num_output_channels=num_output_channels, conv_dims_i=conv_dims_i, conv_pads_i=conv_pads_i, conv_strides_i=conv_strides_i, has_bias=False)
  return convolution_name

def dense_block(f, tab, num, num_layers, parent, batch_norm_size, growth_rate):
  name = 'dense_block_%d' % num
  # Parent of the dense_block is the only parent of the first dense_layer.
  layer_num = 1
  parent_name = dense_layer(f, tab, name, layer_num, [parent], batch_norm_size, growth_rate)
  # Parents of following dense_layer are the preceding dense_layers.
  parents = [parent_name]
  for i	in range(2, num_layers + 1):
    parent_name = dense_layer(f, tab, name, i, parents, batch_norm_size, growth_rate)
    parents.append(parent_name)
  return parent_name # Returns the name of the last dense_layer.

def dense_layer(f, tab, name, num, parents, batch_norm_size, growth_rate):
  complete_name = '%s_dense_layer_%d' % (name, num)
  concatenation_name = concatenation(f, tab, complete_name, parents)
  # conv_dims_i=1 specified in "Densely Connected Convolutional Networks"
  # conv_pads_i=0 (pytorch default), conv_strides_i=1 specified in pytorch implementation
  part1_name = conv_block(f, tab, complete_name + '_part_1', [concatenation_name], num_dims=2, num_output_channels=batch_norm_size * growth_rate, conv_dims_i=1, conv_pads_i=0, conv_strides_i=1)
  # conv_dims_i=3 specified in "Densely Connected Convolutional Networks"
  # conv_pads_i=1, conv_stdies_i=1 specified in pytorch implementation
  part2_name = conv_block(f, tab, complete_name + '_part_2', [part1_name], num_dims=2, num_output_channels=growth_rate, conv_dims_i=3, conv_pads_i=1, conv_strides_i=1)
  return part2_name

def transition_layer(f, tab, num, parent, num_output_channels):
  name = 'transition_layer_%d' % num
  # conv_dims_i=1 specified in "Densely Connected Convolutional Networks"
  # conv_pads_i=0 (pytorch default), conv_strides_i=1 specified in pytorch implementation
  conv_block_name = conv_block(f, tab, name, [parent], num_dims=2, num_output_channels=num_output_channels, conv_dims_i=1, conv_pads_i=0, conv_strides_i=1)
  # pool_dims_i=2, pool_stides_i=2, pool_mode='average' specified in "Densely Connected Convolutional Networks"
  pool_name = pool(f, tab, name, [conv_block_name], num_dims=2, pool_dims_i=2, pool_strides_i=2, pool_mode='average')
  return pool_name

def classification_layer(f, tab, parent):
  name = 'classification_layer'
  # TODO: Find a specific value for pool_strides_i
  # pool_dims_i=7, pool_mode='average' specified in "Densely Connected Convolutional Networks"
  pool_name = pool(f, tab, name, [parent], num_dims=2, pool_dims_i=7, pool_strides_i=2, pool_mode='average')
  fc_name = fully_connected(f, tab, name, [pool_name], num_neurons=1000, has_bias=False)
  softmax_name = softmax(f, tab, name, [fc_name])
  target(f, tab, name, [softmax_name], io_buffer='partitioned', shared_data_reader=True)

### Common #####################################################################
# See https://github.com/LLNL/lbann/blob/develop/src/proto/lbann.proto
# for available prototext parameter options.
### Objective function  ########################################################
def objective_function(f, tab):
  f.write(
"""%sobjective_function {
%s%scross_entropy {
%s%s}
%s%sl2_weight_regularization {
%s%s%sscale_factor: 1e-4
%s%s}
%s}
""" % ((tab,)*13))

### Metrics ####################################################################
def metrics(f, tab):
  f.write(
"""%smetric {
%s%scategorical_accuracy {
%s%s}
%s}
%smetric {
%s%stop_k_categorical_accuracy {
%s%s%stop_k: 5
%s%s}
%s}
""" % ((tab,)*15))

### Callbacks ##################################################################
def callbacks(f, tab, checkpoint=True):
  f.write(
"""%scallback { print {} }
%scallback { timer {} }
%scallback {
%s%simcomm {
%s%s%sintermodel_comm_method: "normal"
%s%s%sall_optimizers: true
%s%s}
%s}""" % ((tab,)*14))
  if checkpoint:
    f.write(
"""
%scallback {
%s%scheckpoint {
%s%s%scheckpoint_dir: "ckpt"
%s%s%scheckpoint_epochs: 1
%s%s}
%s}
""" % ((tab,)*12))
  else:
    f.write('\n')

### Activation Layers ##########################################################
def relu(f, tab, name, parents):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  name += '_relu'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%srelu {\n' % tab)
  tab +=2*' '
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

def softmax(f, tab, name, parents):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  name += '_softmax'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sdata_layout: "model_parallel"\n' % tab)
  f.write('%ssoftmax {\n' % tab)
  tab +=2*' '
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

### Regularization Layers ######################################################
def norm(f, tab, name, parents, decay=None, scale_init=None, bias_init=None, epsilon=None):
  if len(parents) > 1:
    raise Exception('norm must have no more than one parent, but was given %d: %s' % (len(parents), str(parents)))
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  name += '_norm'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%sbatch_normalization {\n' % tab)
  tab +=2*' '
  if decay != None:
    f.write('%sdecay: %f\n' % (tab, decay))
  if scale_init != None:
    f.write('%sscale_init: %f\n' % (tab, scale_init))
  if bias_init != None:
    f.write('%sbias_init: %f\n' % (tab, bias_init))
  if epsilon != None:
    f.write('%sepsilon: %f\n' % (tab, epsilon))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

### Input Layers ###############################################################
# input() is a Python built-in function.
def input_layer(f, tab, name, parents, io_buffer=None, shared_data_reader=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  name += '_input'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%sinput {\n' % tab)
  tab +=2*' '
  if io_buffer != None:
    f.write('%sio_buffer: "%s"\n' % (tab, io_buffer))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

### Transform Layers ###########################################################
def pool(f, tab, name, parents, num_dims=None, pool_dims_i=None, pool_strides_i=None, pool_mode=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  name += '_pool'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%spooling {\n' % tab)
  tab +=2*' '
  if num_dims != None:
      f.write('%snum_dims: %d\n' % (tab, num_dims))
  if pool_dims_i != None:
    f.write('%spool_dims_i: %d\n' % (tab, pool_dims_i))
  if pool_strides_i != None:
    f.write('%spool_strides_i: %d\n' % (tab, pool_strides_i))
  if pool_mode != None:
    f.write('%spool_mode: "%s"\n' % (tab, pool_mode))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

def concatenation(f, tab, name, parents):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  name += '_concatenation'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%sconcatenation {\n' % tab)
  tab +=2*' '
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

### Learning Layers ############################################################
def fully_connected(f, tab, name, parents, num_neurons=None, has_bias=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  name += '_fc'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sdata_layout: "model_parallel"\n' % tab)
  f.write('%sfully_connected {\n' % tab)
  tab +=2*' '
  if num_neurons != None:
    f.write('%snum_neurons: %d\n' % (tab, num_neurons))
  if has_bias != None:
    f.write('%shas_bias: %s\n' % (tab, str(has_bias).lower()))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

def convolution(f, tab, name, parents, num_dims=None, num_output_channels=None,
  conv_dims_i=None, conv_pads_i=None, conv_strides_i=None, has_bias=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  name += '_conv'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%sconvolution {\n' % tab)
  tab +=2*' '
  if num_dims != None:
      f.write('%snum_dims: %d\n' % (tab, num_dims))
  if num_output_channels != None:
      f.write('%snum_output_channels: %d\n' % (tab, num_output_channels))
  if conv_dims_i != None:
    f.write('%sconv_dims_i: %d\n' % (tab, conv_dims_i))
  if conv_pads_i != None:
    f.write('%sconv_pads_i: %d\n' % (tab, conv_pads_i))
  if conv_strides_i != None:
    f.write('%sconv_strides_i: %d\n' % (tab, conv_strides_i))
  if has_bias != None:
    f.write('%shas_bias: %s\n' % (tab, str(has_bias).lower()))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

### Target Layers ##############################################################
def target(f, tab, name, parents, io_buffer=None, shared_data_reader=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  name += '_target'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%starget {\n' % tab)
  tab +=2*' '
  if io_buffer != None:
    f.write('%sio_buffer: "%s"\n' % (tab, io_buffer))
  if shared_data_reader != None:
    f.write('%sshared_data_reader: %s\n' % (tab, str(shared_data_reader).lower()))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

### MAIN ######################################################################
if __name__ == '__main__':
  dense_net(prototext_file='model_densenet_121.prototext', version=121)
  dense_net(prototext_file='model_densenet_161.prototext', version=161)

# Locally, from top-level LBANN directory:
# salloc --nodes=8 --partition=pbatch --time=600
# export MV2_USE_CUDA=1

# srun build/gnu.Release.catalyst.llnl.gov/lbann/build/model_zoo/lbann --reader=/usr/workspace/wsb/forsyth2/lbann/model_zoo/data_readers/data_reader_imagenet.prototext --data_reader_percent=0.010000 --exit_after_setup --model=/usr/workspace/wsb/forsyth2/lbann/model_zoo/models/densenet/model_densenet_121.prototext --num_epochs=5 --optimizer=/usr/workspace/wsb/forsyth2/lbann/model_zoo/optimizers/opt_sgd.prototext > /usr/workspace/wsb/forsyth2/lbann/bamboo/integration_tests/output/densenet_exe_output.txt 2> /usr/workspace/wsb/forsyth2/lbann/bamboo/integration_tests/error/densenet_exe_error.txt

# srun build/gnu.Release.pascal.llnl.gov/lbann/build/model_zoo/lbann --reader=/usr/workspace/wsb/forsyth2/lbann/model_zoo/data_readers/data_reader_imagenet.prototext --data_reader_percent=0.010000 --exit_after_setup --model=/usr/workspace/wsb/forsyth2/lbann/model_zoo/models/densenet/model_densenet_121.prototext --num_epochs=5 --optimizer=/usr/workspace/wsb/forsyth2/lbann/model_zoo/optimizers/opt_sgd.prototext > /usr/workspace/wsb/forsyth2/lbann/bamboo/integration_tests/output/densenet_exe_output.txt 2> /usr/workspace/wsb/forsyth2/lbann/bamboo/integration_tests/error/densenet_exe_error.txt
