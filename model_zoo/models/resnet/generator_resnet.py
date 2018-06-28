# Generate ResNet prototext

# Note that `--learn_rate` for the sgd optimizer should be set to 0.1 for this model.

import sys
sys.path.insert(0, '../python')
import generator_base

def resnet(prototext_file, version):
  f = open(prototext_file, 'w')
  f.write('model {\n')
  tab = ' '*2
  if version == 18:
    f.write(
"""%sname: "directed_acyclic_graph_model"
%sdata_layout: "data_parallel"
%smini_batch_size: 256
%sblock_size: 256
%snum_epochs: 60
""" % ((tab,)*5))
  elif version in [101, 152]:
    f.write(
"""%sname: "directed_acyclic_graph_model"
%sdata_layout: "data_parallel"
%smini_batch_size: 256
%sblock_size: 256
%snum_epochs: 20
""" % ((tab,)*5))
  else:
    raise Exception('Invalid version %d' % version)
  f.write(
"""%snum_parallel_readers: 0
%sprocs_per_model: 0
%snum_gpus: -1
%s###################################################
%s# Objective function
%s###################################################
""" % ((tab,)*6))
  generator_base.objective_function(f, tab)
  f.write(
"""%s###################################################
%s# Metrics
%s###################################################
""" % ((tab,)*3))
  generator_base.metrics(f, tab)
  f.write(
"""%s###################################################
%s# Callbacks
%s###################################################
""" % ((tab,)*3))
  generator_base.callbacks(f, tab, checkpoint_dir='ckptresnet' + str(version))
  if version == 18:
    f.write(
"""%scallback {
%s%sdrop_fixed_learning_rate {
%s%s%sdrop_epoch: 15
%s%s%sdrop_epoch: 30
%s%s%samt: 0.1
%s%s}
%s}
""" % ((tab,)*15))
    f.write(
"""%scallback {
%s%ssummary {
%s%s%sdir: "."
%s%s%sbatch_interval: 1
%s%s%smat_interval: 25
%s%s}
%s}
""" % ((tab,)*15))    
  f.write(
"""%s###################################################
%s# Layers
%s###################################################
""" % ((tab,)*3))
  num_output_channels_tuple = (256, 512, 1024, 2048)
  if version == 18:
    increments = (2, 2, 2, 2)
    layers_tuple = ([64, 64], [128, 128], [256, 256], [512, 512])
    block_type = 'basic'
  elif version == 101:
    increments = (3, 4, 23, 3)
    layers_tuple = ([64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048])
    block_type = 'bottleneck'
  elif version == 152:
    increments = (3, 8, 36, 3)
    layers_tuple = ([64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048])
    block_type = 'bottleneck'
  else:
    raise Exception('Invalid version %d' % version)
  (input_name, parent, index_num) = initial_layer(f, tab)
  resid_num = 1
  num_layers = 4
  for layer_num in range(1, num_layers+1):
    if block_type == 'basic':
      (parent, index_num) = basic_layer(f, tab, index_num, parent, layer_num, increments[layer_num-1], layers_tuple[layer_num-1])
    elif block_type == 'bottleneck':
      (resid_name, index_num) = add_resid(f, tab, resid_num, index_num, [parent], num_output_channels_tuple[layer_num-1])
      (parent, index_num) = bottleneck_layer(f, tab, index_num, parent, resid_name, layer_num, increments[layer_num-1], layers_tuple[layer_num-1])
    else:
      raise Exception('Invalid block_type %s' % block_type)
    resid_num += increments[layer_num-1]
  final_layer(f, tab, index_num, [parent], input_name)
  f.write('}\n')

def initial_layer(f, tab):
  name = 'initial_layer'
  index_num = 0
  parent = 'index_0_%s_input' % name
  (input_name, index_num) = generator_base.input_layer(f, tab, name, index_num, [parent], io_buffer='partitioned')
  (conv_block_name, index_num) = conv_block(f, tab, name, index_num, [input_name], num_dims=2, num_output_channels=64, conv_dims_i=7, conv_pads_i=3, conv_strides_i=2)
  (relu_name, index_num) = generator_base.relu(f, tab, name, index_num, [conv_block_name])
  (pool_name, index_num) = generator_base.pool(f, tab, name, index_num, [relu_name], num_dims=2, pool_dims_i=3, pool_pads_i=1, pool_strides_i=2, pool_mode='max')
  return (input_name, pool_name, index_num)

def add_resid(f, tab, resid_num, index_num, parents, num_output_channels):
  name = 'resid_%d' % resid_num
  if resid_num == 1:
    strides = 1
  else:
    strides = 2
  (conv_block_name, index_num) = conv_block(f, tab, name, index_num, parents, num_dims=2, num_output_channels=num_output_channels, conv_dims_i=1, conv_pads_i=0, conv_strides_i=strides)
  return (conv_block_name, index_num)

def conv_block(f, tab, name, index_num, parents, num_dims=None, num_output_channels=None, conv_dims_i=None, conv_pads_i=None, conv_strides_i=None):
  (convolution_name, index_num) = generator_base.convolution(f, tab, name, index_num, parents, num_dims=num_dims, num_output_channels=num_output_channels, conv_dims_i=conv_dims_i, conv_pads_i=conv_pads_i, conv_strides_i=conv_strides_i, has_bias=False)
  (norm_name, index_num) = generator_base.norm(f, tab, name, index_num, [convolution_name], decay='0.9', scale_init='1.0', bias_init='0.0', epsilon='1e-5')
  return (norm_name, index_num)

def basic_layer(f, tab, index_num, parent, layer_num, num_blocks, layers):
  for block_num in range(1, num_blocks+1):
    name = 'basic_layer_%d' % (layer_num)
    f.write('\n')
    f.write(tab + '###################################################\n')
    f.write(tab + '# Basic Layer %s Block %s\n' % (layer_num, block_num))
    f.write(tab + '###################################################\n')
    f.write('\n\n')
    (parent, index_num) = basic_block(f, tab, name, index_num, parent, layer_num, block_num, layers)
  return (parent, index_num)

def basic_block(f, tab, name, index_num, parent, layer_num, block_num, layers):
  complete_name = '%s_basic_block_%d' % (name, block_num)
  if layer_num != 1 and block_num == 1:
    strides = 2
  else:
    strides = 1
  (conv_block_1_name, index_num) = conv_block(f, tab, complete_name + '_part_1', index_num, [parent], num_dims=2, num_output_channels=layers[0], conv_dims_i=3, conv_pads_i=1, conv_strides_i=strides)
  (relu_1_name, index_num) = generator_base.relu(f, tab, complete_name + '_part_1', index_num, [conv_block_1_name])
  (conv_block_2_name, index_num) = conv_block(f, tab, complete_name + '_part_2', index_num, [relu_1_name], num_dims=2, num_output_channels=layers[1], conv_dims_i=3, conv_pads_i=1, conv_strides_i=1)
  # block 1 of layers 2, 3, and 4 did not have `parent` as a parent in the original prototext.
  if layer_num > 1 and block_num == 1:
    sum_parents = [conv_block_2_name]
  else:
    sum_parents = [parent, conv_block_2_name]
  (sum_name, index_num) = generator_base.sum_layer(f, tab, complete_name, index_num, sum_parents)
  (relu_2_name, index_num) = generator_base.relu(f, tab, complete_name + '_part_2', index_num, [sum_name])
  return (relu_2_name, index_num)

def bottleneck_layer(f, tab, index_num, parent, resid_name, layer_num, num_blocks, layers):
  for block_num in range(1, num_blocks+1):
    name = 'bottleneck_layer_%d' % (layer_num)
    f.write('\n')
    f.write(tab + '###################################################\n')
    f.write(tab + '# BottleNeck Layer %s Block %s\n' % (layer_num, block_num))
    f.write(tab + '###################################################\n')
    f.write('\n\n')
    (parent, index_num) = bottleneck_block(f, tab, name, index_num, parent, resid_name, layer_num, block_num, layers)
  return (parent, index_num)
  
def bottleneck_block(f, tab, name, index_num, parent, resid_name, layer_num, block_num, layers):
  complete_name = '%s_bottleneck_block_%d' % (name, block_num)
  if layer_num != 1 and block_num == 1:
    strides = 2
  else:
    strides = 1
  (conv_block_1_name, index_num) = conv_block(f, tab, complete_name + '_part_1', index_num, [parent], num_dims=2, num_output_channels=layers[0], conv_dims_i=1, conv_pads_i=0, conv_strides_i=strides)
  (relu_1_name, index_num) = generator_base.relu(f, tab, complete_name + '_part_1', index_num, [conv_block_1_name])
  (conv_block_2_name, index_num) = conv_block(f, tab, complete_name + '_part_2', index_num, [relu_1_name], num_dims=2, num_output_channels=layers[1], conv_dims_i=3, conv_pads_i=1, conv_strides_i=1)
  (relu_2_name, index_num) = generator_base.relu(f, tab, complete_name + '_part_2', index_num, [conv_block_2_name])
  (conv_block_3_name, index_num) = conv_block(f, tab, complete_name + '_part_3', index_num, [relu_2_name], num_dims=2, num_output_channels=layers[2], conv_dims_i=1, conv_pads_i=0, conv_strides_i=1)
  sum_parents = [resid_name, conv_block_3_name]
  (sum_name, index_num) = generator_base.sum_layer(f, tab, complete_name, index_num, sum_parents)
  (relu_3_name, index_num) = generator_base.relu(f, tab, complete_name + '_part_3', index_num, [sum_name])
  return (relu_3_name, index_num)

def final_layer(f, tab, index_num, parents, input_name):
  name = 'final_layer'
  (pool_name, index_num) = generator_base.pool(f, tab, name, index_num, parents, num_dims=2, pool_dims_i=7, pool_pads_i=0, pool_strides_i=1, pool_mode="average")
  (fully_connected_name, index_num) = generator_base.fully_connected(f, tab, name, index_num, [pool_name], num_neurons=1000, has_bias=False)
  (softmax_name, index_num) = generator_base.softmax(f, tab, name, index_num, [fully_connected_name])
  generator_base.target(f, tab, name, index_num, [softmax_name, input_name], io_buffer='partitioned', shared_data_reader=True)

if __name__ == '__main__':
  # Currently set to sequential for diffs - remove that part for merging
  resnet(prototext_file='resnet18/model_resnet18_sequential.prototext', version=18)
  resnet(prototext_file='resnet101/model_resnet101.prototext', version=101)
  resnet(prototext_file='resnet152/model_resnet152.prototext', version=152)
