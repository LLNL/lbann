import numpy as np

# Data paths
data_dir = '/p/lustre2/brainusr/datasets/zinc/moses_zinc_train250K.npy'
samples = np.load(data_dir, allow_pickle=True)

dims = len(samples[0])


pad_indx = 28
# Sample access functions
def get_sample(index):
    sample = samples[index]
    if len(sample) < dims:
        sample = np.concatenate((sample, np.full(dims-len(sample), pad_indx)))
    else:
        sample = np.resize(sample, dims)
    return sample

def num_samples():
    return samples.shape[0]

def sample_dims():
    return [dims]

def str_list(l):
    return ' '.join([str(i) for i in l])
# ==============================================
# Setup and launch experiment
# ==============================================

def construct_model():
    """Construct LBANN model.

    Initial model for ATOM molecular SMILES generation
    Network architecture and training hyperparameters from
    https://github.com/samadejacobs/moses/tree/master/moses/char_rnn

    """
    import lbann
    import lbann.modules

    sequence_length = sample_dims()[0]
    data_layout = 'data_parallel'

    # Layer graph
    input = lbann.Input(name='inp_tensor')
    x_slice = lbann.Slice(
        input,
        axis=0,
        slice_points=str_list(range(sequence_length+1)),
        device='CPU',
        name='inp_slice'
    )

    #embedding layer
    emb = []
    embedding_size=30
    dictionary_size=30

    emb_weights = lbann.Weights(
        initializer=lbann.NormalInitializer(mean=0, standard_deviation=1),
        name='emb_matrix'
    )

    lstm1 = lbann.modules.GRU(size=768, data_layout=data_layout)
    fc = lbann.modules.FullyConnectedModule(size=dictionary_size, data_layout=data_layout)


    last_output = lbann.Constant(value=0.0,
                                 num_neurons='768',
                                 data_layout=data_layout,
                                 name='lstm_init_output')

    lstm1_prev_state = [last_output]


    gt  = lbann.Constant(value=0, num_neurons='57')
    loss= []
    idl = []
    for i in range(sequence_length):
      idl.append(lbann.Identity(x_slice, name='slice_idl_'+str(i), device='CPU'))

    for i in range(sequence_length-1):
        emb_l = lbann.Embedding(
            idl[i],
            num_embeddings=dictionary_size,
            embedding_dim=embedding_size,
            name='emb_'+str(i),
            device='CPU',
            weights=emb_weights
        )

        x,lstm1_prev_state = lstm1(emb_l,lstm1_prev_state)
        fc_l = fc(x)
        y_soft = lbann.Softmax(fc_l, name='soft_'+str(i))
        gt = lbann.OneHot(idl[i+1], size=dictionary_size)
        ce = lbann.CrossEntropy([y_soft,gt],name='loss_'+str(i))
        #mask padding in input
        pad_mask = lbann.NotEqual([idl[i],lbann.Constant(value=pad_indx,num_neurons='1')],device='CPU')
        ce_mask = lbann.Multiply([pad_mask,ce],name='loss_mask_'+str(i))
        loss.append(lbann.LayerTerm(ce_mask, scale=1/(sequence_length-1)))


    layers = list(lbann.traverse_layer_graph(input))
    # Setup objective function
    weights = set()
    for l in layers:
      weights.update(l.weights)
    obj = lbann.ObjectiveFunction(loss)


    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 lbann.CallbackStepLearningRate(step=10, amt=0.5),
                 lbann.CallbackDumpWeights(basename="weights",epoch_interval=50)]

    # Construct model
    mini_batch_size = 64
    num_epochs = 50
    return lbann.Model(mini_batch_size,
                       num_epochs,
                       weights=weights,
                       layers=layers,
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
    data_reader.percent_of_data_to_use = 1.0
    data_reader.python.module = module_name
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    return message

if __name__ == '__main__':
    import lbann
    import lbann.contrib.launcher
    trainer = lbann.Trainer()
    model = construct_model()
    opt = lbann.Adam(learn_rate=0.001,beta1=0.9,beta2=0.99,eps=1e-8)
    data_reader = construct_data_reader()
    status = lbann.contrib.launcher.run(
        trainer, model, data_reader, opt,
        account='hpcdl',
        scheduler='slurm',
        time_limit=720,
        nodes=1,
        job_name='atom_char_rnn_250K')
    print(status)
