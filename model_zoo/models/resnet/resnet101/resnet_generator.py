
def res_net101(prototext_file='model_resnet.prototext', version=101):
    f = open(prototext_file, 'w')
    f.write('model {\n')
    tab = ' '*2
    f.write(tab + 'name: "sequential_model"\n')
    f.write(tab + '###################################################\n')
    f.write(tab + '# Layers\n')
    f.write(tab + '###################################################\n')
    input_layer(f, tab)
    initial_layer(f, tab)
    num = 1
    add_resid(f, tab, num, "initial_layer_pool", 256)
    bottleneck_layer(f, tab, num, 3, [64, 64, 256])
    num += 3
    add_resid(f, tab, num, "bottleneck_block_%d_3_relu" % (num-1), 512)
    bottleneck_layer(f, tab, num, 4, [128, 128, 512])
    num += 4
    add_resid(f, tab, num, "bottleneck_block_%d_3_relu" % (num-1), 1024)
    bottleneck_layer(f, tab, num, 23, [256, 256, 1024])
    num += 23
    add_resid(f, tab, num, "bottleneck_block_%d_3_relu" % (num-1), 2048)
    bottleneck_layer(f, tab, num, 3, [512, 512, 2048])
    num += 3
    final_layer(f, tab, num)
    softmax(f, tab, "final_layer", "final_layer_fc")
    output_layer(f, tab, "final_layer_softmax")

    f.write('}\n')


def res_net152(prototext_file='model_resnet152.prototext', version=152):
    f = open(prototext_file, 'w')
    f.write('model {\n')
    tab = ' '*2
    f.write(tab + 'name: "sequential_model"\n')
    f.write(tab + '###################################################\n')
    f.write(tab + '# Layers\n')
    f.write(tab + '###################################################\n')
    input_layer(f, tab)
    initial_layer(f, tab)
    num = 1
    add_resid(f, tab, num, "initial_layer_pool", 256)
    bottleneck_layer(f, tab, num, 3, [64, 64, 256])
    num += 3
    add_resid(f, tab, num, "bottleneck_block_%d_3_relu" % (num-1), 512)
    bottleneck_layer(f, tab, num, 8, [128, 128, 512])
    num += 8
    add_resid(f, tab, num, "bottleneck_block_%d_3_relu" % (num-1), 1024)
    bottleneck_layer(f, tab, num, 36, [256, 256, 1024])
    num += 36
    add_resid(f, tab, num, "bottleneck_block_%d_3_relu" % (num-1), 2048)
    bottleneck_layer(f, tab, num, 3, [512, 512, 2048])
    num += 3
    final_layer(f, tab, num)
    softmax(f, tab, "final_layer", "final_layer_fc")
    output_layer(f, tab, "final_layer_softmax")

    f.write('}\n')


def bottleneck_layer(f, tab, num, repeat, layers):
    for i in range(repeat):
        name = 'bottleneck_block_%d' % (num)
        f.write(tab + '###################################################\n')
        f.write(tab + '# BottleNeck %s Block %s\n' % (num, i+1))
        f.write(tab + '###################################################\n')
        bottleneck_block(f, tab, name, num, layers, i)
        num +=1

def add_resid(f, tab, num, parents, num_output_channels):
    conv(f, tab, "resid_%d" % num, parents, 2, num_output_channels, 1, 0, 1)

def bottleneck_block(f, tab, name, num, layers, i):
    if i == 0:
        parents = ['resid_%d_norm' % num, 'bottleneck_block_%d_bottleneck_conv_%d_part_3_norm' % (num, num)]
    else:
        parents = ['bottleneck_block_%d_3_relu' % (num - 1), 'bottleneck_block_%d_bottleneck_conv_%d_part_3_norm' % (num, num)]
    #else:
    #    parents = ['bottleneck_block_%d_bottleneck_conv_%d_part_3_norm' % (num, num), ""]
    if num == 1:
        cp1 = "initial_layer_pool"
    else:
        cp1 = 'bottleneck_block_%d_3_relu' % (num-1)
    complete_name = '%s_bottleneck_conv_%d_part_' % (name, num)
    conv(f, tab, complete_name + str(1), cp1, 2, layers[0], 1, 0, 1)
    relu_1_parent = complete_name + '1_norm'
    relu(f, tab, name + '_' + str(1), relu_1_parent)
    conv_2_parent = name + '_1_relu'
    conv(f, tab, complete_name + str(2), conv_2_parent, 2, layers[1], 3, 1, 1)
    relu_2_parent = complete_name + '2_norm'
    relu(f, tab, name + '_' + str(2), relu_2_parent)
    conv_3_parent = name + '_2_relu'
    conv(f, tab, complete_name + str(3), conv_3_parent, 2, layers[2], 1, 0 ,1)
    sum_layer(f, tab, name, parents)
    relu_3_parent = name + '_sum'
    relu(f, tab, name + '_' + str(3), relu_3_parent)


def conv(f, tab, name, parents,num_dims, num_output_channels, conv_dims_i, conv_pads_i, conv_strides_i):
    convolution(f, tab, name, parents, num_dims, num_output_channels, conv_dims_i, conv_pads_i, conv_strides_i)
    norm_parents = name+'_conv'
    norm(f, tab, name, norm_parents)




#def initial_layer(f, tab):
#  name = 'initial_layer'
#  conv(f, tab, name, conv_dims_i=7, conv_strides_i=2)
#  pool(f, tab, name, pool_dims_i=3, pool_strides_i=2, pool_mode='max')

def input_layer(f, tab):
    f.write('%slayer {\n' % tab)
    tab += 2*' '
    f.write('%sname: "0"\n' % tab)
    f.write('%sparents: "0"\n' % tab)
    f.write('%sdata_layout: "data_parallel"\n' % tab)
    f.write('%sinput {\n' % tab)
    tab += 2 * ' '
    f.write('%sio_buffer: "partitioned"\n' % tab)
    tab = tab[:-2]
    f.write('%s}\n' % tab)
    tab = tab[:-2]
    f.write('%s}\n' % tab)

def output_layer(f, tab, parents):
    f.write('%slayer {\n' % tab)
    tab += 2*' '
    f.write('%sname: "output_layer"\n' % tab)
    f.write('%sparents: "%s"\n' % (tab, parents))
    f.write('%sdata_layout: "data_parallel"\n' % tab)
    f.write('%starget {\n' % tab)
    tab += 2 * ' '
    f.write('%sio_buffer: "partitioned"\n' % tab)
    f.write('%sshared_data_reader: true\n' % tab)
    tab = tab[:-2]
    f.write('%s}\n' % tab)
    tab = tab[:-2]
    f.write('%s}\n' % tab)


def initial_layer(f, tab):
    name = 'initial_layer'
    convolution(f, tab, name, "0", num_dims=2, num_output_channels=64, conv_dims_i=7, conv_pads_i=3, conv_strides_i=2)
    norm(f, tab, name, name+'_conv', decay=0.9, scale_init=1, bias_init=0.0, epsilon="1e-5")
    relu(f, tab, name, name+'_norm')
    pool(f, tab, name, name+'_relu', num_dims=2, pool_dims_i=3, pool_pads_i=1, pool_strides_i=2, pool_mode="max")

def final_layer(f, tab, num):
    name = 'final_layer'
    pool_parent = 'bottleneck_block_%d_3_relu' % (num - 1)
    pool(f, tab, name, pool_parent, num_dims=2, pool_dims_i=7, pool_pads_i=0, pool_strides_i=1, pool_mode="average")
    fully_connected(f, tab, name, 'final_layer_pool', num_neurons=1000, has_bias=False)

def transition_layer(f, tab, num):
  name = 'transition_layer_%d' % num
  conv(f, tab, name, conv_dims_i=1)
  pool(f, tab, name, pool_dims_i=2, pool_strides_i=2, pool_mode='average')

def classification_layer(f, tab):
  name = 'classification_layer'
  pool(f, tab, name, pool_dims_i=7, pool_mode='average')
  fully_connected(f, tab, name, num_neurons=1000, has_bias=False)

### Common Layers #############################################################

#def conv(f, tab, name, conv_dims_i=None, conv_strides_i=None):
#  norm(f, tab, name)
#  relu(f, tab, name)
#  convolution(f, tab, name, conv_dims_i=conv_dims_i, conv_strides_i=conv_strides_i)

def norm(f, tab, name, parents, decay=0.9, scale_init=1.0, bias_init=0.0, epsilon="1e-5"):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  f.write('%sname: "%s_norm"\n' % (tab, name))
  f.write('%sparents: "%s"\n' % (tab, parents))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%sbatch_normalization {\n' % tab)
  tab += 2*' '
  f.write('%sdecay: %.1f\n' % (tab, decay))
  f.write('%sscale_init: %.1f\n' % (tab, scale_init))
  f.write('%sbias_init: %.1f\n' % (tab, bias_init))
  f.write('%sepsilon: %s\n' % (tab, epsilon))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)

def relu(f, tab, name, parent):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  f.write('%sname: "%s_relu"\n' % (tab, name))
  f.write('%sparents: "%s"\n' % (tab, parent))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%srelu {\n' % tab)
  tab = tab + (2*' ')
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)

def convolution(f, tab, name, parent, num_dims=None, num_output_channels=None, conv_dims_i=None, conv_pads_i= None, conv_strides_i=None, has_bias="false"):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  f.write('%sname: "%s_conv"\n' % (tab, name))
  f.write('%sparents: "%s"\n' % (tab, parent))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%sconvolution {\n' % tab)
  tab += 2 * ' '
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
  f.write('%shas_bias: %s\n' % (tab, has_bias))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)

def pool(f, tab, name, parents, num_dims=0, pool_dims_i=None, pool_pads_i=0, pool_strides_i=None, pool_mode=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  f.write('%sname: "%s_pool"\n' % (tab, name))
  f.write('%sparents: "%s"\n' % (tab, parents))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%spooling {\n' % tab)
  tab += 2*' '
  f.write('%snum_dims: %d\n' % (tab, num_dims))
  if pool_dims_i != None:
    f.write('%spool_dims_i: %d\n' % (tab, pool_dims_i))
  if pool_strides_i != None:
    f.write('%spool_strides_i: %d\n' % (tab, pool_strides_i))
  if pool_pads_i != None:
    f.write('%spool_pads_i: %d\n' % (tab, pool_pads_i))
  if pool_mode != None:
    f.write('%spool_mode: "%s"\n' % (tab, pool_mode))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)

def fully_connected(f, tab, name, parent, num_neurons=None, has_bias=None):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  f.write('%sname: "%s_fc"\n' % (tab, name))
  f.write('%sparents: "%s"\n' % (tab, parent))
  f.write('%sdata_layout: "model_parallel"\n' % tab)
  f.write('%sfully_connected {\n' % tab)
  tab += 2*' '
  if num_neurons != None:
    f.write('%snum_neurons: %d\n' % (tab, num_neurons))
  if has_bias != None:
    f.write('%shas_bias: %s\n' % (tab, str(has_bias)))
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)

def sum_layer(f, tab, name, parents):
    f.write('%slayer {\n' % tab)
    tab += 2*' '
    f.write('%sname: "%s_sum"\n' % (tab, name))
    f.write('%sparents: "%s %s"\n' % (tab, parents[0], parents[1]))
    f.write('%sdata_layout: "data_parallel"\n' % tab)
    f.write('%ssum {\n' % tab)
    f.write('%s}\n' % tab )
    tab = tab[:-2]
    f.write('%s}\n' % tab)

def softmax(f, tab, name, parents):
  f.write('%slayer {\n' % tab)
  tab += 2*' '
  f.write('%sname: "%s_softmax"\n' % (tab, name))
  f.write('%sparents: "%s"\n' % (tab, parents))
  f.write('%sdata_layout: "data_parallel"\n' % tab)
  f.write('%ssoftmax {\n' % tab)
  tab += 2*' '
  tab = tab[:-2]
  f.write('%s}\n' % tab)
  tab = tab[:-2]
  f.write('%s}\n' % tab)
### MAIN ######################################################################

if __name__ == '__main__':
  res_net152()
