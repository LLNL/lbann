import sys
import os
import subprocess
import functools

# Parameters
lbann_dir       = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip()
lbann_proto_dir = lbann_dir + '/src/proto/'
work_dir        = lbann_dir + '/model_zoo/models/gan/jags/cycle_gan'
template_proto  = lbann_dir + '/model_zoo/models/gan/jags/cycle_gan/cycgan_m1_template.prototext'
output_proto    = lbann_dir + '/model_zoo/models/gan/jags/cycle_gan/cycgan_m1.prototext'

# Convert a list into a space-separated string
def str_list(l):
    if isinstance(l, list):
        return ' '.join(str(i) for i in l)
    elif isinstance(l, str):
        return l
    else:
        raise TypeError('str_list expects a list or a string')

# Construct a new layer and add it to the model
def new_layer(model, name, parents, layer_type, layout = 'data_parallel'):
    l = model.layer.add()
    l.name = name
    l.data_layout = layout
    l.parents = str_list(parents)
    #l.device_allocation = device
    exec('l.' + layer_type + '.SetInParent()')
    return l

# Construct a new set of weights and add it to the model
def new_weights(model, name, initializer = 'constant_initializer'):
    w = model.weights.add()
    w.name = name
    exec('w.' + initializer + '.SetInParent()')
    return w

# Discriminator
#@todo: clean up, tag may not be needed
#Weight sharing on the same branch (D1) or (D2)
def add_discriminator(model,disc_input, prefix, freeze=False, add_weight=True, tag=''):
  #Shared weights for same path (e.g. D1 fake and D1 real)
  w1 = prefix+'fc1'
  w2 = prefix+'fc2'
  w3 = prefix+'fc3'

  fc1 = w1+tag
  fc2 = w2+tag
  fc3 = w3+tag


  relu1 = prefix+'relu1'+tag
  relu2 = prefix+'relu2'+tag

  l = new_layer(model, fc1, disc_input,'fully_connected')
  l.fully_connected.num_neurons = 128
  l.fully_connected.has_bias = True
  l.freeze = freeze
  if(add_weight) :
    w = new_weights(model, w1 + 'linearity', 'he_normal_initializer')
  l.weights = w1 + 'linearity'

  l = new_layer(model, relu1, fc1,'relu')
  

  l = new_layer(model, fc2, relu1,'fully_connected')
  l.fully_connected.num_neurons = 16
  l.fully_connected.has_bias = True
  l.freeze = freeze
  if(add_weight) :
    w = new_weights(model, w2 + 'linearity', 'he_normal_initializer')
  l.weights = w2 + 'linearity'
  
  l = new_layer(model, relu2, fc2,'relu')

  l = new_layer(model, fc3, relu2, 'fully_connected')
  l.fully_connected.num_neurons = 1
  l.fully_connected.has_bias = True
  l.freeze = freeze
  if(add_weight) :
    w = new_weights(model, w3 + 'linearity', 'he_normal_initializer')
  l.weights = w3 + 'linearity'
  return fc3 


#Generator
#Weight frozen, no weight sharing
#todo, handle weight sharing
def add_generator(model, gen_input, prefix, output_dim, freeze=False, add_dropout=True, tag=''):
  #different weights  
  fc1 = prefix+'fc1'+tag
  fc2 = prefix+'fc2'+tag
  fc3 = prefix+'fc3'+tag
  fc4 = prefix+'fc4'+tag

  relu1 = prefix+'relu1'+tag
  relu2 = prefix+'relu2'+tag
  relu3 = prefix+'relu3'+tag

  dropout1 = prefix+'dropout1'+tag

  l = new_layer(model, fc1, gen_input,'fully_connected')
  l.fully_connected.num_neurons = 16
  l.fully_connected.has_bias = True
  l.freeze = freeze
  w = new_weights(model, fc1 + 'linearity', 'he_normal_initializer')
  l.weights = fc1 + 'linearity'

  l = new_layer(model, relu1, fc1,'relu')

  l = new_layer(model, fc2, relu1,'fully_connected')
  l.fully_connected.num_neurons = 128
  l.fully_connected.has_bias = True
  l.freeze = freeze
  w = new_weights(model, fc2 + 'linearity', 'he_normal_initializer')
  l.weights = fc2 + 'linearity'
  
  l = new_layer(model, relu2, fc2,'relu')
  next_parent = relu2
  if(add_dropout):
    l = new_layer(model,dropout1,next_parent, 'dropout')
    l.dropout.keep_prob = 0.8
    next_parent=dropout1

  l = new_layer(model, fc3, next_parent, 'fully_connected')
  l.fully_connected.num_neurons = 512
  l.fully_connected.has_bias = True
  l.freeze = freeze
  w = new_weights(model, fc3 + 'linearity', 'he_normal_initializer')
  l.weights = fc3 + 'linearity'
  
  l = new_layer(model, relu3, fc3, 'relu')

  l = new_layer(model, fc4, relu3, 'fully_connected')
  l.fully_connected.num_neurons = output_dim
  l.fully_connected.has_bias = True
  l.freeze = freeze
  w = new_weights(model, fc4 + 'linearity', 'he_normal_initializer')
  l.weights = fc4 + 'linearity'

  return fc4


# Configure a prototext model (e.g. add layers)
def configure_model(model):

    #####INPUT DATA (including Slices)
    ### Input data comes from merge features of image (Y) and param (X)
    l = new_layer(model,'data',' ', 'input')
    l.input.io_buffer = 'partitioned'
    
    slice_points = [0,2500,2511]
    l = new_layer(model, 'slice_data','data', 'slice')
    l.children = 'image_data_dummy param_data_id'
    l.slice.slice_points = str_list(slice_points)

    #ID Image (Y) data
    l = new_layer(model,'image_data_dummy','slice_data','identity')

    #ID parameter data (X)
    l = new_layer(model,'param_data_id','slice_data','identity')
    
    #D_Loss1 branch
    #Fake path
    #def add_generator(model, gen_input, prefix, output_dim, freeze=False, add_dropout=True, tag=''):
    #freeze generator = True
    #g_sample=generator1(x)
    g_sample = add_generator(model, 'param_data_id','gen1', 2500, True,True)
    #g_sample2= generator(y)
    g_sample2 = add_generator(model,'image_data_dummy','gen2', 11, True,False)
    
    #True path (share weights with fake path discriminator)
    #discriminator(y,x)
    #data = y + x
    D_real = add_discriminator(model, 'data','disc1',False, True, '_real')
    #CONCAT 
    # Gsample + x
    #
    l = new_layer(model, 'concat_gsample_n_param','','concatenation')
    l.parents = g_sample+' param_data_id'
    #discriminator false path
    #question: how to deal with weight sharing?
    #discriminator(g_sample,x)
    D_fake = add_discriminator(model,'concat_gsample_n_param','disc1',False, False, '_fake')
    
    #obectives here (D_real, D_fake)

    #D_loss2 branch
    #Reconcatenate x+y
    l = new_layer(model, 'concat_param_n_img','param_data_id image_data_dummy','concatenation')
  
    #D_real2 = discriminator2(x,y)
    D_real2 = add_discriminator(model,'concat_param_n_img','disc2',False, True, '_real')
    
    #D_fake2  = discriminator2(G_sample2,y)
    l = new_layer(model, 'concat_gsample2_n_img',g_sample2+ ' image_data_dummy','concatenation')
    D_fake2 = add_discriminator(model,'concat_gsample2_n_img','disc2', False, False, '_fake')

    #Objective and evaluation layers here
    l = new_layer(model, 'disc1_real_bce', D_real, 'bce_with_logits')
    l.bce_with_logits.true_label = 1
    l = new_layer(model, 'disc1_real_eval','disc1_real_bce', 'evaluation')

    l = new_layer(model, 'disc1_fake_bce', D_fake, 'bce_with_logits')
    l.bce_with_logits.true_label = int(0)
    l = new_layer(model, 'disc1_fake_eval','disc1_fake_bce', 'evaluation')
 
    l = new_layer(model, 'disc2_real_bce', D_real2, 'bce_with_logits')
    l.bce_with_logits.true_label = 1
    l = new_layer(model, 'disc2_real_eval','disc2_real_bce', 'evaluation')

    l = new_layer(model, 'disc2_fake_bce', D_fake2, 'bce_with_logits')
    l.bce_with_logits.true_label = int(0)
    l = new_layer(model, 'disc2_fake_eval','disc2_fake_bce', 'evaluation')


if __name__ == "__main__":

    # Make sure protobuf Python implementation is built
    host = subprocess.check_output('hostname').strip('\n1234567890')
    protoc = lbann_dir + '/build/gnu.' + host + '.llnl.gov/install/bin/protoc'
    proto_python_dir = lbann_dir + '/build/gnu.' + host + '.llnl.gov/protobuf/src/python'
    os.putenv('PROTOC', protoc)
    subprocess.call('cd ' + proto_python_dir + '; '
                    + sys.executable + ' '
                    + proto_python_dir + '/setup.py build',
                    shell=True)
    sys.path.append(proto_python_dir)
    import google.protobuf.text_format as txtf

    # Compile LBANN protobuf
    subprocess.call([protoc,
                     '-I=' + lbann_proto_dir,
                     '--python_out=' + work_dir,
                     lbann_proto_dir + '/lbann.proto'])
    sys.path.append(work_dir)
    global lbann_pb2
    import lbann_pb2

    # Load template prototext
    with open(template_proto, 'r') as f:
        pb = txtf.Merge(f.read(), lbann_pb2.LbannPB())

    # Configure prototext model
    configure_model(pb.model)

    # Export prototext
    with open(output_proto, 'w') as f:
        f.write(txtf.MessageToString(pb))
    
