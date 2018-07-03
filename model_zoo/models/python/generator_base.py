# Shared functions (i.e. base file) to write prototext for models.
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
def callbacks(f, tab, checkpoint_dir=None):
  if checkpoint_dir != None:
    f.write(
"""%scallback {
%s%scheckpoint {
%s%s%scheckpoint_dir: "%s"
""" % (tab, tab, tab, tab, tab, tab, checkpoint_dir))
    f.write(
"""%s%s%scheckpoint_epochs: 1
%s%s}
%s}
""" % ((tab,)*6))
  f.write(
"""%scallback { print {interval: 1} }
%scallback { timer {} }
%scallback {
%s%simcomm {
%s%s%sintermodel_comm_method: "normal"
%s%s%sall_optimizers: true
%s%s}
%s}
""" % ((tab,)*14))

### Activation Layers ##########################################################
def relu(f, tab, name, parents):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  name += '_relu'
  f.write('%sname: "%s"\n' % (tab, name))
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
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
  name += '_softmax'
  f.write('%sname: "%s"\n' % (tab, name))
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
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
  name += '_norm'
  f.write('%sname: "%s"\n' % (tab, name))
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%sbatch_normalization {\n' % tab)
  tab +=2*' '
  if decay != None:
    f.write('%sdecay: %s\n' % (tab, decay))
  if scale_init != None:
    f.write('%sscale_init: %s\n' % (tab, scale_init))
  if bias_init != None:
    f.write('%sbias_init: %s\n' % (tab, bias_init))
  if epsilon != None:
    f.write('%sepsilon: %s\n' % (tab, epsilon))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

### Input Layers ###############################################################
# input() is a Python built-in function, so this function cannot have that name.
def input_layer(f, tab, name, parents, io_buffer=None, shared_data_reader=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  name += '_input'
  f.write('%sname: "%s"\n' % (tab, name))
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
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
def pool(f, tab, name, parents, num_dims=None, pool_dims_i=None, pool_pads_i=None, pool_strides_i=None, pool_mode=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  name += '_pool'
  f.write('%sname: "%s"\n' % (tab, name))
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%spooling {\n' % tab)
  tab +=2*' '
  if num_dims != None:
      f.write('%snum_dims: %d\n' % (tab, num_dims))
  if pool_dims_i != None:
    f.write('%spool_dims_i: %d\n' % (tab, pool_dims_i))
  if pool_pads_i != None:
    f.write('%spool_pads_i: %d\n' % (tab, pool_pads_i))
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
  name += '_concatenation'
  f.write('%sname: "%s"\n' % (tab, name))
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%sconcatenation {\n' % tab)
  tab +=2*' '
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

# sum() is a Python built-in function, so this function cannot have that name. 
def sum_layer(f, tab, name, parents):
  if len(parents) != 2:
    raise Exception('sum must have exactly two parents, but was given %d: %s' % (len(parents), str(parents)))
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  name += '_sum'
  f.write('%sname: "%s"\n' % (tab, name))
  f.write('%sparents: "%s %s"\n' % (tab, parents[0], parents[1]))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%ssum {\n' % tab)
  tab +=2*' '
  tab = tab[:-2]
  f.write('%s}\n' % tab )
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  return name

### Learning Layers ############################################################
def fully_connected(f, tab, name, parents, num_neurons=None, has_bias=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  name += '_fc'
  f.write('%sname: "%s"\n' % (tab, name))
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
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
  name += '_conv'
  f.write('%sname: "%s"\n' % (tab, name))
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
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
  name += '_target'
  f.write('%sname: "%s"\n' % (tab, name))
  if parents != []:
    f.write('%sparents: "%s"\n' % (tab, ' '.join(parents)))
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
