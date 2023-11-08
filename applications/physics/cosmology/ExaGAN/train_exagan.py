import ExaGAN
import dataset
import lbann.contrib.launcher

# ==============================================
# Setup and launch experiment
# ==============================================

def list2str(l):
    return ' '.join(l)

def construct_model():
    """Construct LBANN model.

    ExaGAN  model

    """
    import lbann

    # Layer graph
    input = lbann.Input(data_field='samples',name='inp_img')
    #label flipping
    label_flip_rand = lbann.Uniform(min=0,max=1, neuron_dims=[1])
    label_flip_prob = lbann.Constant(value=0.01, num_neurons=[1])
    one = lbann.GreaterEqual(label_flip_rand,label_flip_prob, name='is_real')
    zero = lbann.LogicalNot(one,name='is_fake')

    z = lbann.Reshape(lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims=[64], name='noise_vec'),dims=[1, 64])
    d1_real, d1_fake, d_adv, gen_img  = ExaGAN.CosmoGAN()(input,z)

    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,one],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zero],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,one],name='d_adv_bce')

    layers = list(lbann.traverse_layer_graph(input))
    # Setup objective function
    weights = set()
    src_layers = []
    dst_layers = []
    for l in layers:
      if(l.weights and "disc1" in l.name and "instance1" in l.name):
        src_layers.append(l.name)
      #freeze weights in disc2, analogous to discrim.trainable=False in Keras
      if(l.weights and "disc2" in l.name):
        dst_layers.append(l.name)
        for idx in range(len(l.weights)):
          l.weights[idx].optimizer = lbann.NoOptimizer()
      weights.update(l.weights)
    #l2_reg = lbann.L2WeightRegularization(weights=weights, scale=1e-4)
    obj = lbann.ObjectiveFunction([d1_real_bce,d1_fake_bce,d_adv_bce])
    # Initialize check metric callback
    metrics = [lbann.Metric(d1_real_bce,name='d_real'),
               lbann.Metric(d1_fake_bce, name='d_fake'),
               lbann.Metric(d_adv_bce,name='gen')]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 #Uncomment to dump output for plotting and further statistical analysis
                 #lbann.CallbackDumpOutputs(layers='inp_img gen_img_instance1_activation',
                 #                          execution_modes='train validation',
                 #                          directory='dump_outs',
                 #                          batch_interval=100,
                 #                          format='npy'),
                 lbann.CallbackReplaceWeights(source_layers=list2str(src_layers),
                                      destination_layers=list2str(dst_layers),
                                      batch_interval=2)]

    # Construct model
    num_epochs = 20
    return lbann.Model(num_epochs,
                       weights=weights,
                       layers=layers,
                       metrics=metrics,
                       objective_function=obj,
                       callbacks=callbacks)

def construct_data_reader():
    """Construct Protobuf message for Python data reader.

    The Python data reader will import this Python file to access the
    sample access functions.

    """
    import os.path
    import lbann
    module_file = os.path.abspath(__file__)
    module_name = os.path.splitext(os.path.basename(module_file))[0]
    module_dir = os.path.dirname(module_file)

    # Base data reader message
    message = lbann.reader_pb2.DataReader()

    # Training set data reader
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'train'
    data_reader.shuffle = True
    data_reader.fraction_of_data_to_use = 1.0
    data_reader.validation_fraction = 0.1
    data_reader.python.module = 'dataset'
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    return message

if __name__ == '__main__':
    import lbann

    mini_batch_size = 64
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
    model = construct_model()
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0002,beta1=0.5,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader = construct_data_reader()
    status = lbann.contrib.launcher.run(trainer,model, data_reader, opt,
                       scheduler='slurm',
                       #account='lbpm',
                       nodes=1,
                       procs_per_node=1,
                       time_limit=1440,
                       setup_only=False,
                       job_name='exagan')
    print(status)
