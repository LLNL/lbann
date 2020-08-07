import numpy as np
import models.vae as molvae
#from misc import KLAnnealer

# Data paths
#data_dir = '/usr/workspace/wsa/jacobs32/deepL/moses.fork/scripts/moses_zinc_train250K.npy'
#data_dir = '/p/lustre2/brainusr/datasets/zinc/moses_zinc_train250K.npy'
#data_dir = '/p/gscratchr/brainusr/datasets/zinc/moses_zinc_train250K.npy'
data_dir = '/p/gpfs1/brainusr/datasets/zinc/moses_zinc_train250K.npy'
train_samples = np.load(data_dir)[:200000] #200K for training
val_samples = np.load(data_dir)[200000:250000] #1000 sampes for validation

#samples = np.load(data_dir)
print("DATA DIR ", data_dir)
print("TRAIN DATA_SHAPE ", train_samples.shape)

print("VAL DATA_SHAPE ", val_samples.shape)

dims = len(train_samples[0])
#dims = 30 #hack to test nan
pad_indx = 27 #from pytorch
print("DIMS ", dims)
# Sample access functions
def get_sample(index):
    #np.random.shuffle(train_samples)
    sample = train_samples[index]
    diff = dims - len(sample)
    sample = np.concatenate((sample,np.full(diff,pad_indx))) if diff > 0 else np.resize(sample,dims)
    return sample

def num_samples():
    return train_samples.shape[0]

def sample_dims():
    return [dims]

def get_val_sample(index):
    val_sample = val_samples[index]
    diff = dims - len(val_sample)
    val_sample = np.concatenate((val_sample,np.full(diff,pad_indx))) if diff > 0 else np.resize(val_sample,dims)
    return val_sample

def num_val_samples():
    return val_samples.shape[0]

def str_list(l):
    return ' '.join([str(i) for i in l])
# ==============================================
# Setup and launch experiment
# ==============================================

def construct_model():
    """Construct LBANN model.

    Initial model for ATOM molecular VAE 

    """
    import lbann

    # Layer graph
    input = lbann.Input(name='inp_tensor')
    #Slice input layer feature dims (axis=0)
    vae_loss= []
    x_slice = lbann.Slice(input, axis=0, slice_points=str_list(list(range(sample_dims()[0]+1))),name='inp_slice')
    input_feature_dims = sample_dims()[0]
    dictionary_size = 29 #from pytorch
    embedding_size = 29 #from pytorch
    kl, recon,arg_max = molvae.MolVAE(input_feature_dims,dictionary_size,
                                embedding_size, pad_indx)(x_slice)

    vae_loss.extend(kl)
    #vae_loss.extend(recon[:-1])
    recon_loss=[]
    #kl_weight_loss=[]
    #kl_weight = lbann.Weights(initializer=lbann.ConstantInitializer())
    for i in range(input_feature_dims-1):
      recon_loss.append(lbann.LayerTerm(recon[i],scale=1/(input_feature_dims-1))) 
      '''
      kl_weight_loss.append(lbann.Multiply(
                       lbann.WeightsLayer(weights=kl_weight, dims='1'),
                               kl_loss[i],
                               ))
      loss = lbann.Add(
                 lbann.Multiply(
                       lbann.WeightsLayer(weights=kl_weight, dims='1'),
                               kl_loss[i],
                               ),
                        #recon_loss[:-1],
                        lbann.LayerTerm(recon[i],scale=1/(input_feature_dims-1)) 
                    )
      recon_loss.append(loss)
      '''
    #vae_loss.extend(kl_weight_loss)
    vae_loss.extend(recon_loss)
    #vae_loss.extend(recon_loss)
    print("LEN vae loss ", len(vae_loss))
    #vae_loss.extend(recon[:-1])
    #metric layers
    kl_metric = lbann.Reduction(lbann.Concatenation(kl),mode='average')
    recon_metric = lbann.Reduction(lbann.Concatenation(recon[:-1]), mode='average')
    pred_tensor = lbann.Concatenation(arg_max[:-1], name='pred_tensor')

    layers = list(lbann.traverse_layer_graph(input))
    # Setup objective function
    weights = set()
    for l in layers:
      weights.update(l.weights)
    #l2_reg = lbann.L2WeightRegularization(weights=weights, scale=5e-4)
    #kl_weight = lbann.Weights(lbann.ConstantInitializer(123.0))
    '''
    loss = lbann.Add(
                 lbann.Multiply(
                       lbann.WeightsLayer(weights=kl_weight, dims='1'),
                               kl_loss,
                               ),
                        recon_loss[:-1],
                    )
    '''
    #vae_loss.extend(kl_weight_loss)
    obj = lbann.ObjectiveFunction(vae_loss)
    #obj = lbann.ObjectiveFunction(recon_loss)
    #obj = lbann.ObjectiveFunction(kl_weight_loss)

    # Initialize check metric callback
    metrics = [lbann.Metric(kl_metric, name='kl_loss'),
               lbann.Metric(recon_metric, name='recon')
                ]
    num_epochs = 100

    #kl_annealer = KLAnnealer(num_epochs, self.config)
    '''
    kl_annealer = KLAnnealer(num_epochs)

    kl_callbacks = []
    for epoch in range(0, num_epochs):
      kl_callbacks.append(
         lbann.CallbackSetWeightsValue(
              weight_name=kl_weight.name,
              weight_value=kl_annealer(epoch),
              epoch_interval=epoch,
          )
       )
    '''
    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 #lbann.CallbackStepLearningRate(step=10, amt=0.5),
                 #lbann.CallbackDumpGradients(basename='grads',interval=100),
                 lbann.CallbackDumpWeights(directory='weights', epoch_interval=4)]
    #callbacks.extend(kl_callbacks)
                  

    # Construct model
    return lbann.Model(num_epochs,
                       weights=weights,
                       layers=layers,
                       objective_function=obj,
                       metrics=metrics,
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
    # TODO: This can be removed once
    # https://github.com/LLNL/lbann/issues/1098 is resolved.
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'train'
    data_reader.shuffle = True
    data_reader.percent_of_data_to_use = 1.0
    data_reader.python.module = module_name
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    # Test set data reader
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'validate'
    data_reader.percent_of_data_to_use = 1.0
    data_reader.python.module = module_name
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'get_val_sample'
    data_reader.python.num_samples_function = 'num_val_samples'
    data_reader.python.sample_dims_function = 'sample_dims'
    return message

if __name__ == '__main__':
    import lbann
    import lbann.contrib.launcher
    model = construct_model()
    #opt = lbann.Optimizer() # No optimizer
    # Setup optimizer
    #opt = lbann.SGD(learn_rate=0.01, momentum=0.9)
    opt = lbann.Adam(learn_rate=0.0003,beta1=0.9,beta2=0.99,eps=1e-8)
    data_reader = construct_data_reader()
    mini_batch_size = 512
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size,
        name=None,
       # procs_per_trainer=run_args.procs_per_trainer,
    )
    status = lbann.contrib.launcher.run(trainer,model, data_reader, opt,
                       scheduler='lsf',
                       #account='hpcdl',
                       nodes=1,
                       procs_per_node=4,
                       batch_job = True,
                       time_limit=720,
                       setup_only = False,
                       job_name='go_vae_shuffleT_emb29')
    print(status)
