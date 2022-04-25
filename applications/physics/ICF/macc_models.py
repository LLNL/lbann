import lbann
import macc_network_architectures

def list2str(l):
    return ' '.join(l)

def construct_jag_wae_model(ydim,
                            zdim,
                            mcf,
                            useCNN,
                            dump_models,
                            ltfb_batch_interval,
                            num_epochs
                            ):
    """Construct LBANN model.

    JAG Wasserstein autoencoder  model

    """

    # Layer graph
    input = lbann.Input(data_field='samples', name='inp_data')
    # data is 64*64*4 images + 15 scalar + 5 param
    #inp_slice = lbann.Slice(input, axis=0, slice_points=[0, 16399, 16404],name='inp_slice')
    inp_slice = lbann.Slice(input, axis=0, slice_points=[0,ydim,ydim+5],name='inp_slice')
    gt_y = lbann.Identity(inp_slice,name='gt_y')
    gt_x = lbann.Identity(inp_slice, name='gt_x') #param not used

    zero  = lbann.Constant(value=0.0,num_neurons=[1],name='zero')
    one  = lbann.Constant(value=1.0,num_neurons=[1],name='one')

    z_dim = 20  #Latent space dim

    z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims=20)
    model = macc_network_architectures.MACCWAE(zdim,ydim,cf=mcf,use_CNN=useCNN)
    d1_real, d1_fake, d_adv, pred_y  = model(z,gt_y)

    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,one],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zero],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,one],name='d_adv_bce')
    img_loss = lbann.MeanSquaredError([pred_y,gt_y])
    rec_error = lbann.L2Norm2(lbann.WeightedSum([pred_y,gt_y], scaling_factors=[1, -1]))

    layers = list(lbann.traverse_layer_graph(input))
    # Setup objective function
    weights = set()
    src_layers = []
    dst_layers = []
    for l in layers:
      if(l.weights and "disc0" in l.name and "instance1" in l.name):
        src_layers.append(l.name)
      #freeze weights in disc2
      if(l.weights and "disc1" in l.name):
        dst_layers.append(l.name)
        for idx in range(len(l.weights)):
          l.weights[idx].optimizer = lbann.NoOptimizer()
      weights.update(l.weights)
    l2_reg = lbann.L2WeightRegularization(weights=weights, scale=1e-4)
    d_adv_bce = lbann.LayerTerm(d_adv_bce,scale=0.01)
    obj = lbann.ObjectiveFunction([d1_real_bce,d1_fake_bce,d_adv_bce,img_loss,rec_error,l2_reg])
    # Initialize check metric callback
    metrics = [lbann.Metric(img_loss, name='recon_error')]
    #pred_y = macc_models.MACCWAE.pred_y_name
    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 lbann.CallbackPrintModelDescription(),
                 lbann.CallbackSaveModel(dir=dump_models),
                 lbann.CallbackReplaceWeights(source_layers=list2str(src_layers),
                                      destination_layers=list2str(dst_layers),
                                      batch_interval=2)]

    if(ltfb_batch_interval > 0) :
      callbacks.append(lbann.CallbackLTFB(batch_interval=ltfb_batch_interval,metric='recon_error',
                                    low_score_wins=True,
                                    exchange_hyperparameters=True))

    # Construct model
    return lbann.Model(num_epochs,
                       weights=weights,
                       layers=layers,
                       metrics=metrics,
                       objective_function=obj,
                       callbacks=callbacks)

def construct_macc_surrogate_model(xdim,
                                   ydim,
                                   zdim,
                                   wae_mcf,
                                   surrogate_mcf,
                                   lambda_cyc,
                                   useCNN,
                                   dump_models,
                                   pretrained_dir,
                                   ltfb_batch_interval,
                                   num_epochs
                                   ):
    """Construct MACC surrogate model.

    See https://arxiv.org/pdf/1912.08113.pdf model architecture and other details

    """
    # Layer graph
    input = lbann.Input(data_field='samples',name='inp_data')
    # data is 64*64*4 images + 15 scalar + 5 param
    inp_slice = lbann.Slice(input, axis=0, slice_points=[0,ydim,ydim+xdim],name='inp_slice')
    gt_y = lbann.Identity(inp_slice,name='gt_y')
    gt_x = lbann.Identity(inp_slice, name='gt_x') #param not used

    zero  = lbann.Constant(value=0.0,num_neurons=[1],name='zero')
    one  = lbann.Constant(value=1.0,num_neurons=[1],name='one')


    z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims=20)
    wae = macc_network_architectures.MACCWAE(zdim,ydim,cf=wae_mcf,use_CNN=useCNN) #pretrained, freeze
    inv = macc_network_architectures.MACCInverse(xdim,cf=surrogate_mcf)
    fwd = macc_network_architectures.MACCForward(zdim,cf=surrogate_mcf)


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
    loss_gen0  = lbann.WeightedSum([L_l2_y,L_cyc], scaling_factors=[1, lambda_cyc])
    loss_gen1  = lbann.WeightedSum([L_l2_x,L_cyc_y], scaling_factors=[1, lambda_cyc])
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

    l2_reg = lbann.L2WeightRegularization(weights=weights, scale=1e-4)
    #d_adv_bce = lbann.LayerTerm(d_adv_bce,scale=0.01)
    # Setup objective function
    obj = lbann.ObjectiveFunction([loss_gen0,loss_gen1,l2_reg])
    # Initialize check metric callback
    metrics = [lbann.Metric(img_sca_loss, name='fw_loss'),
               lbann.Metric(L_l2_x, name='inverse loss'),
               lbann.Metric(L_cyc_y, name='output cycle loss'),
               lbann.Metric(L_cyc_x, name='param cycle loss')]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackSaveModel(dir=dump_models),
                 lbann.CallbackLoadModel(dirs=str(pretrained_dir)),
                 lbann.CallbackTimer()]

    if(ltfb_batch_interval > 0) :
      callbacks.append(lbann.CallbackLTFB(batch_interval=ltfb_batch_interval,metric='fw_loss',
                                    low_score_wins=True,
                                    exchange_hyperparameters=True))
    # Construct model
    return lbann.Model(num_epochs,
                       weights=weights,
                       layers=layers,
                       metrics=metrics,
                       objective_function=obj,
                       callbacks=callbacks)
