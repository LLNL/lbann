import lbann
import models.wae as molwae
from lbann.util import list2str

def construct_atom_wae_model(pad_index,
                             sequence_length,
                             embedding_size,
                             dictionary_size,
                             z_dim,
                             g_mean,
                             g_std,
                             lambda_,
                             lr,
                             batch_size,
                             num_epochs,
                             dump_outputs_dir = None,
                             dump_weights_dir = None,
                             dump_weights_interval = 0,
                             dump_model_dir = None,
                             dump_outputs_interval = 0,
                             warmup = False,
                             CPU_only = False):
    """Construct LBANN ATOM WAE model.

    Initial model for ATOM molecular VAE

    """
    import lbann

    assert pad_index is not None
    assert sequence_length is not None

    print("sequence length is {}".format(sequence_length))
    data_layout = "data_parallel"
    # Layer graph
    input_ = lbann.Input(name='inp',data_field='samples')
    input_feature_dims = sequence_length

    assert embedding_size is not None
    assert dictionary_size is not None

    save_output = True if dump_outputs_dir else False

    print("save output? ", save_output, "out dir ",  dump_outputs_dir)
    z = lbann.Gaussian(mean=g_mean,stdev=g_std, neuron_dims=str(z_dim))
    recon, d1_real, d1_fake, d_adv, arg_max  = molwae.MolWAE(
        input_feature_dims,
        dictionary_size,
        embedding_size,
        pad_index,
        z_dim,
        g_mean,
        g_std,
        save_output=save_output)(input_,z)


    zero  = lbann.Constant(value=0.0,num_neurons=[1],name='zero')
    one  = lbann.Constant(value=1.0,num_neurons=[1],name='one')

    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,one],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zero],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,one],name='d_adv_bce')

    #vae_loss.append(recon)

    layers = list(lbann.traverse_layer_graph(input_))

    # Hack to avoid non-deterministic floating-point errors in some
    # GPU layers
    if(CPU_only):
        for l in layers:
            if isinstance(l, lbann.Embedding) or isinstance(l, lbann.Tessellate):
                l.device = 'CPU'

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
    l2_weights = [w for w in weights if not isinstance(w.optimizer, lbann.NoOptimizer)]
    l2_reg = lbann.L2WeightRegularization(weights=l2_weights, scale=1e-4)

    d_adv_bce = lbann.LayerTerm(d_adv_bce,scale=lambda_)

    obj = lbann.ObjectiveFunction([d1_real_bce,d1_fake_bce,d_adv_bce,recon,l2_reg])

    # Initialize check metric callback
    metrics = [
               lbann.Metric(recon, name='recon')
                ]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer()]

    if(dump_weights_interval > 0):
      callbacks.append(lbann.CallbackDumpWeights(directory=dump_weights_dir,
                                              epoch_interval=dump_weights_interval))

    callbacks.append(lbann.CallbackReplaceWeights(source_layers=list2str(src_layers),
                                 destination_layers=list2str(dst_layers),
                                 batch_interval=2))

    #Dump final weight for inference
    if(dump_model_dir):
      callbacks.append(lbann.CallbackSaveModel(dir=dump_model_dir))

    #Dump output (activation) for post processing
    if(dump_outputs_dir):
      pred_tensor = lbann.Concatenation(arg_max, name='pred_tensor')
      callbacks.append(lbann.CallbackDumpOutputs(batch_interval=dump_outputs_interval,
                       execution_modes='test', directory=dump_outputs_dir,layers='inp pred_tensor'))

    if(warmup):
        callbacks.append(
            lbann.CallbackLinearGrowthLearningRate(
                target=lr / 512 * batch_size, num_epochs=5))

    # Construct model
    return lbann.Model(num_epochs,
                       weights=weights,
                       layers=layers,
                       objective_function=obj,
                       metrics=metrics,
                       callbacks=callbacks)
