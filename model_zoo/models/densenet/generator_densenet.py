# Generate DenseNet prototext
# See https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# See "Densely Connected Convolutional Networks" by Huang et. al

# Note that `--learn_rate` for the sgd optimizer should be set to 0.1 for this model.

import sys
sys.path.insert(0, '../python')
import generator_base

def densenet(prototext_file, version):
  f = open(prototext_file, 'w')
  f.write('model {\n')
  tab = ' '*2
  format_tuple = (tab,)*11
  f.write(
"""%sname: "directed_acyclic_graph_model"
%sdata_layout: "data_parallel"
%smini_batch_size: 256
%sblock_size: 256
%snum_epochs: 90
%snum_parallel_readers: 0
%sprocs_per_model: 0
%snum_gpus: -1
%s###################################################
%s# Objective function
%s###################################################
""" % ((tab,)*11))
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
  generator_base.callbacks(f, tab, checkpoint_dir='ckptdensenet' + str(version))
  f.write(
"""%scallback {
%s%sdrop_fixed_learning_rate {
%s%s%sdrop_epoch: 30
%s%s%sdrop_epoch: 60
%s%s%samt: 0.1
%s%s}
%s}
""" % ((tab,)*15))
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
  (input_name, parent, index_num) = initial_layer(f, tab, number_initial_features)
  num_blocks = 4
  for num in range(1, num_blocks + 1):
    num_layers = num_layers_tuple[num - 1]
    (parent, index_num) = dense_block(f, tab, num, num_layers, index_num, parent, number_initial_features, batch_norm_size, growth_rate)
    if num != num_blocks:
      transition_num_features = number_initial_features + (num_layers * growth_rate)
      (parent, index_num) = transition_layer(f, tab, num, index_num, parent, transition_num_features / 2.0)
  classification_layer(f, tab, index_num, parent, input_name)
  f.write('}\n')

def initial_layer(f, tab, num_initial_channels):
  name = 'initial_layer'
  index_num = 0
  (input_name, index_num) = generator_base.input_layer(f, tab, name, index_num, [], io_buffer='partitioned')
  # conv_dims_i=7, conv_strides_i=2 specified in "Densely Connected Convolutional Networks"
  # conv_pads_i=3 specified in pytorch implementation
  (conv_block_name, index_num) = conv_block_initial(f, tab, name, index_num, [input_name], num_dims=2, num_output_channels=num_initial_channels, conv_dims_i=7, conv_pads_i=3, conv_strides_i=2)
  # pool_dims_i=3, pool_strides_i=2, pool_mode='max' specified in "Densely Connected Convolutional Networks"
  # pool_pads_i=1 recommended
  (pool_name, index_num) = generator_base.pool(f, tab, name, index_num, [conv_block_name], num_dims=2, pool_dims_i=3, pool_pads_i=1, pool_strides_i=2, pool_mode='max')
  return (input_name, pool_name, index_num)

def conv_block_initial(f, tab, name, index_num, parents, num_dims=None, num_output_channels=None, conv_dims_i=None, conv_pads_i=None, conv_strides_i=None):
  (convolution_name, index_num) = generator_base.convolution(f, tab, name, index_num, parents, num_dims=num_dims, num_output_channels=num_output_channels, conv_dims_i=conv_dims_i, conv_pads_i=conv_pads_i, conv_strides_i=conv_strides_i, has_bias=False)
  (norm_name, index_num) = generator_base.norm(f, tab, name, index_num, [convolution_name], decay='0.9', scale_init='1.0', bias_init='0.0', epsilon='1e-5')
  (relu_name, index_num) = generator_base.relu(f, tab, name, index_num, [norm_name])
  return (relu_name, index_num)

def dense_block(f, tab, num, num_layers, index_num, parent, num_initial_channels, batch_norm_size, growth_rate):
  name = 'dense_block_%d' % num
  num_output_channels = num_initial_channels*(2**(num-1)) # Double for each block
  layer_num = 1
  # Only parent of the first dense_layer is the dense_block's parent.
  (parent_name, index_num) = dense_layer(f, tab, name, layer_num, index_num, [parent], batch_norm_size, growth_rate, num_output_channels + ((layer_num-1)*growth_rate))
  # Parents of following dense_layer are the preceding dense_layers.
  parents = [parent_name]
  for layer_num	in range(2, num_layers + 1):
    (parent_name, index_num) = dense_layer(f, tab, name, layer_num, index_num, parents, batch_norm_size, growth_rate, num_output_channels + ((layer_num-1)*growth_rate))
    parents.append(parent_name)
  # parent_name is the name of the last dense_layer.
  return (parent_name, index_num)

def dense_layer(f, tab, name, num, index_num, parents, batch_norm_size, growth_rate, num_output_channels):
  complete_name = '%s_dense_layer_%d' % (name, num)
  (concatenation_name, index_num) = generator_base.concatenation(f, tab, complete_name, index_num, parents)
  # conv_dims_i=1 specified in "Densely Connected Convolutional Networks"
  # conv_pads_i=0 (pytorch default), conv_strides_i=1 specified in pytorch implementation
  (part1_name, index_num) = conv_block(f, tab, complete_name + '_part_1', index_num, [concatenation_name], num_dims=2, num_output_channels=batch_norm_size * growth_rate, conv_dims_i=1, conv_pads_i=0, conv_strides_i=1)
  # conv_dims_i=3 specified in "Densely Connected Convolutional Networks"
  # conv_pads_i=1, conv_strides_i=1 specified in pytorch implementation
  (part2_name, index_num) = conv_block(f, tab, complete_name + '_part_2', index_num, [part1_name], num_dims=2, num_output_channels=num_output_channels, conv_dims_i=3, conv_pads_i=1, conv_strides_i=1)
  return (part2_name, index_num)

def conv_block(f, tab, name, index_num, parents, num_dims=None, num_output_channels=None, conv_dims_i=None, conv_pads_i=None, conv_strides_i=None):
  (norm_name, index_num) = generator_base.norm(f, tab, name, index_num, parents, decay=0.9, scale_init=1.0, bias_init=0.0, epsilon=1e-5)
  (relu_name, index_num) = generator_base.relu(f, tab, name, index_num, [norm_name])
  (convolution_name, index_num) = generator_base.convolution(f, tab, name, index_num, [relu_name], num_dims=num_dims, num_output_channels=num_output_channels, conv_dims_i=conv_dims_i, conv_pads_i=conv_pads_i, conv_strides_i=conv_strides_i, has_bias=False)
  return (convolution_name, index_num)

def transition_layer(f, tab, num, index_num, parent, num_output_channels):
  name = 'transition_layer_%d' % num
  # conv_dims_i=1 specified in "Densely Connected Convolutional Networks"
  # conv_pads_i=0 (pytorch default), conv_strides_i=1 specified in pytorch implementation
  (conv_block_name, index_num) = conv_block(f, tab, name, index_num, [parent], num_dims=2, num_output_channels=num_output_channels, conv_dims_i=1, conv_pads_i=0, conv_strides_i=1)
  # pool_dims_i=2, pool_strides_i=2, pool_mode='average' specified in "Densely Connected Convolutional Networks"
  (pool_name, index_num) = generator_base.pool(f, tab, name, index_num, [conv_block_name], num_dims=2, pool_dims_i=2, pool_strides_i=2, pool_mode='average')
  return (pool_name, index_num)

def classification_layer(f, tab, index_num, parent, input_name):
  name = 'classification_layer'
  # pool_dims_i=7, pool_mode='average' specified in "Densely Connected Convolutional Networks"
  (pool_name, index_num) = generator_base.pool(f, tab, name, index_num, [parent], num_dims=2, pool_dims_i=7, pool_strides_i=1, pool_mode='average')
  (fc_name, index_num) = generator_base.fully_connected(f, tab, name, index_num, [pool_name], num_neurons=1000, has_bias=False)
  (softmax_name, index_num) = generator_base.softmax(f, tab, name, index_num, [fc_name])
  generator_base.target(f, tab, name, index_num, [softmax_name, input_name], io_buffer='partitioned', shared_data_reader=True)

### MAIN ######################################################################
if __name__ == '__main__':
  densenet(prototext_file='model_densenet_121.prototext', version=121)
  densenet(prototext_file='model_densenet_161.prototext', version=161)

# Locally, from top-level LBANN directory:
# salloc --nodes=32 --partition=pbatch --time=600
# export MV2_USE_CUDA=1

# srun build/gnu.Release.catalyst.llnl.gov/lbann/build/model_zoo/lbann --reader=/usr/workspace/wsb/forsyth2/lbann/model_zoo/data_readers/data_reader_imagenet.prototext --data_reader_percent=0.010000 --exit_after_setup --model=/usr/workspace/wsb/forsyth2/lbann/model_zoo/models/densenet/model_densenet_121.prototext --num_epochs=5 --optimizer=/usr/workspace/wsb/forsyth2/lbann/model_zoo/optimizers/opt_sgd.prototext > /usr/workspace/wsb/forsyth2/lbann/bamboo/integration_tests/output/densenet_exe_output.txt 2> /usr/workspace/wsb/forsyth2/lbann/bamboo/integration_tests/error/densenet_exe_error.txt

# srun build/gnu.Release.pascal.llnl.gov/lbann/build/model_zoo/lbann --reader=/usr/workspace/wsb/forsyth2/lbann/model_zoo/data_readers/data_reader_imagenet.prototext --data_reader_percent=0.010000 --exit_after_setup --model=/usr/workspace/wsb/forsyth2/lbann/model_zoo/models/densenet/model_densenet_121.prototext --num_epochs=5 --optimizer=/usr/workspace/wsb/forsyth2/lbann/model_zoo/optimizers/opt_sgd.prototext > /usr/workspace/wsb/forsyth2/lbann/bamboo/integration_tests/output/densenet_exe_output.txt 2> /usr/workspace/wsb/forsyth2/lbann/bamboo/integration_tests/error/densenet_exe_error.txt
