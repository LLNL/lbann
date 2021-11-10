import lbann
import macc_network_architectures
from lbann.util import str_list, list2str

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
    #inp_slice = lbann.Slice(input, axis=0, slice_points="0 16399 16404",name='inp_slice')
    inp_slice = lbann.Slice(input, axis=0, slice_points=str_list([0,ydim,ydim+5]),name='inp_slice')
    gt_y = lbann.Identity(inp_slice,name='gt_y')
    gt_x = lbann.Identity(inp_slice, name='gt_x') #param not used

    zero  = lbann.Constant(value=0.0,num_neurons='1',name='zero')
    one  = lbann.Constant(value=1.0,num_neurons='1',name='one')

    z_dim = 20  #Latent space dim

    z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="20")
    model = macc_network_architectures.MACCWAE(zdim,ydim,cf=mcf,use_CNN=useCNN)
    d1_real, d1_fake, d_adv, pred_y  = model(z,gt_y)

    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,one],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zero],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,one],name='d_adv_bce')
    img_loss = lbann.MeanSquaredError([pred_y,gt_y])
    rec_error = lbann.L2Norm2(lbann.WeightedSum([pred_y,gt_y], scaling_factors="1 -1"))

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
