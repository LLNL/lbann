import macc_models 
from os.path import abspath, dirname, join
import google.protobuf.text_format as txtf
import lbann.contrib.lc.launcher

# ==============================================
# Setup and launch experiment
# ==============================================

# Default data reader
cur_dir = dirname(abspath(__file__))
data_reader_prototext = join(dirname(cur_dir),
                             'data',
                             'jag_100Kdata.prototext')

model_dir=''
#Load at least pretrained WAE model
assert model_dir, 'pre_trained_dir should not be empty'
#Assume pre_trained model is in current directory, change path if not
pre_trained_dir=join(cur_dir,model_dir) 

def list2str(l):
    return ' '.join(l)

def construct_model():
    """Construct MACC surrogate model.

    See https://arxiv.org/pdf/1912.08113.pdf model architecture and other details

    """
    import lbann

    # Layer graph
    input = lbann.Input(target_mode='N/A',name='inp_data')
    # data is 64*64*4 images + 15 scalar + 5 param
    inp_slice = lbann.Slice(input, axis=0, slice_points="0 16399 16404",name='inp_slice')
    gt_y = lbann.Identity(inp_slice,name='gt_y')
    gt_x = lbann.Identity(inp_slice, name='gt_x') #param not used

    zero  = lbann.Constant(value=0.0,num_neurons='1',name='zero')
    one  = lbann.Constant(value=1.0,num_neurons='1',name='one')

    y_dim = 16399 #image+scalar shape
    z_dim = 20  #Latent space dim
    x_dim = 5
    lamda_cyc = 1e-3

    z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="20")
    wae = macc_models.MACCWAE(z_dim,y_dim) #pretrained, freeze
    inv = macc_models.MACCInverse(x_dim)
    fwd = macc_models.MACCForward(z_dim)
    
    
    y_pred_fwd = wae.encoder(gt_y)

    param_pred_ = wae.encoder(gt_y)
    input_fake = inv(param_pred_)

    output_cyc = fwd(input_fake)
    y_image_re2  = wae.decoder(output_cyc)

    '''**** Train cycleGAN input params <--> latent space of (images, scalars) ****'''
    output_fake = fwd(gt_x)
    y_image_re = wae.decoder(output_fake)

    param_pred2_ = wae.encoder(y_image_re)
    input_cyc = inv(param_pred2_)

    L_l2_x =  lbann.MeanSquaredError(input_fake,gt_x)
    L_cyc_x = lbann.MeanSquaredError(input_cyc,gt_x)

    L_l2_y =  lbann.MeanSquaredError(output_fake,y_pred_fwd)
    L_cyc_y = lbann.MeanSquaredError(output_cyc,y_pred_fwd)
   
     
    #@todo slice here to separate scalar from image
    img_sca_loss = lbann.MeanSquaredError(y_image_re,gt_y)
    #L_cyc = L_cyc_y + L_cyc_x
    L_cyc = lbann.Add(L_cyc_y, L_cyc_x)

    #loss_gen0  = L_l2_y + lamda_cyc*L_cyc
    loss_gen0  = lbann.WeightedSum([L_l2_y,L_cyc], scaling_factors=f'1 {lamda_cyc}')
    loss_gen1  = lbann.WeightedSum([L_l2_x,L_cyc_y], scaling_factors=f'1 {lamda_cyc}')
    #loss_gen1  =  L_l2_x + lamda_cyc*L_cyc_y


    layers = list(lbann.traverse_layer_graph(input))
    weights = set()
    #Freeze appropriate (pretrained) weights
    pretrained_models = ["wae"]  #add macc?
    for l in layers:
      for idx in range(len(pretrained_models)):
        if(l.weights and pretrained_models[idx] in l.name):
          for w in range(len(l.weights)):
            l.weights[w].optimizer = lbann.NoOptimizer()
      weights.update(l.weights)
         
    '''
    l2_reg = lbann.L2WeightRegularization(weights=weights, scale=1e-4)
    d_adv_bce = lbann.LayerTerm(d_adv_bce,scale=0.01)
    '''
    # Setup objective function
    obj = lbann.ObjectiveFunction([loss_gen0,loss_gen1])
    # Initialize check metric callback
    metrics = [lbann.Metric(img_sca_loss, name='img_sca_loss'),
               lbann.Metric(L_l2_x, name='inverse loss'),
               lbann.Metric(L_cyc_y, name='output cycle loss'),
               lbann.Metric(L_cyc_x, name='param cycle loss')]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackLoadModel(dir=str(pre_trained_dir)),
                 lbann.CallbackTimer()]
                                            
    # Construct model
    mini_batch_size = 128
    num_epochs = 100
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       weights=weights,
                       layers=layers,
                       metrics=metrics,
                       objective_function=obj,
                       callbacks=callbacks)


if __name__ == '__main__':
    import lbann
    
    trainer = lbann.Trainer()
    model = construct_model()
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0001,beta1=0.9,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader_proto = lbann.lbann_pb2.LbannPB()
    with open(data_reader_prototext, 'r') as f:
      txtf.Merge(f.read(), data_reader_proto)
    data_reader_proto = data_reader_proto.data_reader

    status = lbann.contrib.lc.launcher.run(trainer,model, data_reader_proto, opt,
                       scheduler='slurm',
                       nodes=1,
                       procs_per_node=1,
                       time_limit=360,
                       setup_only=True, 
                       job_name='macc_surrogate')
    print(status)
