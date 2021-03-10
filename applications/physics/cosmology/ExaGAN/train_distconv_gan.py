import DistConvGAN
import dataset3D
import lbann.contrib.launcher
import lbann.contrib.args
import argparse
from lbann.core.util import get_parallel_strategy_args

# ==============================================
# Setup and launch experiment
# ==============================================

def list2str(l):
    return ' '.join(l)

def construct_lc_launcher_args():
    desc = ('Construct and run  3D CosmoGAN on CosmoFlow (4 channel) dataset.'
            'Single channel dataset is also supported using Python data reader.')
    parser = argparse.ArgumentParser(description=desc)
    lbann.contrib.args.add_scheduler_arguments(parser)

    # General arguments
    parser.add_argument(
        '--job-name', action='store', default='lbann_cosmo3DGAN', type=str,
        help='scheduler job name (default: lbann_cosmoae)')
    parser.add_argument("--scheduler", type=str, default="lsf")
    parser.add_argument(
        '--mini-batch-size', action='store', default=1, type=int,
        help='mini-batch size (default: 1)', metavar='NUM')
    parser.add_argument(
        '--num-nodes', action='store', default=1, type=int,
        help='num-nodes (default: 1)', metavar='NUM')
    parser.add_argument(
        '--num-epochs', action='store', default=5, type=int,
        help='number of epochs (default: 100)', metavar='NUM')
    parser.add_argument(
        '--random-seed', action='store', default=None, type=int,
        help='the random seed (default: None)')

    # Model specific arguments
    parser.add_argument(
        '--input-width', action='store', default=128, type=int,
        help='the input spatial width (default: 128)')
    parser.add_argument(
        '--input-channel', action='store', default=1, type=int,
        help='the input channel (default: 1)')
    parser.add_argument(
        '--num-secrets', action='store', default=4, type=int,
        help='number of secrets (default: 4)')
    parser.add_argument(
        '--use-batchnorm', action='store_true',
        help='Use batch normalization layers')
    parser.add_argument(
        '--local-batchnorm', action='store_true',
        help='Use local batch normalization mode')
    default_lc_dataset = '/p/gpfs1/brainusr/datasets/cosmoflow/cosmoUniverse_2019_05_4parE/hdf5_transposed_dim128_float/batch8'
    for role in ['train', 'val', 'test']:
        default_dir = '{}/{}'.format(default_lc_dataset, role)
        parser.add_argument(
            '--{}-dir'.format(role), action='store', type=str,
            default=default_dir,
            help='the directory of the {} dataset'.format(role))

    # Parallelism arguments
    parser.add_argument(
        '--depth-groups', action='store', type=int, default=4,
        help='the number of processes for the depth dimension (default: 4)')
    parser.add_argument(
        '--depth-splits-pooling-id', action='store', type=int, default=5,
        help='the number of pooling layers from which depth_split is set (default: 5)')
    parser.add_argument(
        '--gather-dropout-id', action='store', type=int, default=1,
        help='the number of dropout layers from which the network is gathered (default: 1)')

    parser.add_argument(
        '--use-python-reader', action='store_true',
        help='Use python data reader instead of HDF5 (default: false)')

    parser.add_argument(
        '--dynamically-reclaim-error-signals', action='store_true',
        help='Allow LBANN to reclaim error signals buffers (default: False)')

    return parser.parse_args()

    # Construct layer graph
def construct_model(run_args):
    """Construct LBANN model.

    ExaGAN  model

    """
    import lbann

    # Layer graph
    input = lbann.Input(target_mode='N/A',name='inp_img')
    #input = lbann.Input(name='input',
    #    target_mode='reconstruction')
    #label flipping
    label_flip_rand = lbann.Uniform(min=0,max=1, neuron_dims='1')
    label_flip_prob = lbann.Constant(value=0.01, num_neurons='1')
    one = lbann.GreaterEqual(label_flip_rand,label_flip_prob, name='is_real')
    zero = lbann.LogicalNot(one,name='is_fake')

    z = lbann.Reshape(lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="64", name='noise_vec'),dims='1 64')
    print("RUN ARGS ", run_args) 
    d1_real, d1_fake, d_adv, gen_img  = DistConvGAN.Exa3DGAN(run_args.input_width,
                                         run_args.input_channel)(input,z)

    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,one],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zero],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,one],name='d_adv_bce')

    layers = list(lbann.traverse_layer_graph(input))
    #Add parallel strategy
    parallel_strategy = get_parallel_strategy_args(
        sample_groups=run_args.mini_batch_size,
        depth_groups=run_args.depth_groups)
   
    '''
    supported_layers=["Input","Convolution", "Deconvolution", "Relu","Tanh", "FullyConnected", "BatchNormalization"]
    for i, layer in enumerate(layers):
        layer_name = layer.__class__.__name__
        if layer_name in supported_layers:
          layer.parallel_strategy = parallel_strategy
    '''

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
    obj = lbann.ObjectiveFunction([d1_real_bce,d1_fake_bce,d_adv_bce])
    # Initialize check metric callback
    metrics = [lbann.Metric(d1_real_bce,name='d_real'),
               lbann.Metric(d1_fake_bce, name='d_fake'),
               lbann.Metric(d_adv_bce,name='gen')]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer(),
                 lbann.CallbackGPUMemoryUsage(),
                 #lbann.CallbackPrintModelDescription(),
                 #lbann.CallbackDebug(),
                 lbann.CallbackProfiler(skip_init=True),
                 lbann.CallbackReplaceWeights(source_layers=list2str(src_layers),
                                      destination_layers=list2str(dst_layers),
                                      batch_interval=2)]

    # Construct model
    return lbann.Model(epochs=run_args.num_epochs,
                       weights=weights,
                       layers=layers,
                       metrics=metrics,
                       objective_function=obj,
                       callbacks=callbacks)


def create_hdf5_data_reader(
        train_path, val_path, test_path, num_responses):
    """Create a data reader for CosmoFlow.

    Args:
        {train, val, test}_path (str): Path to the corresponding dataset.
        num_responses (int): The number of parameters to predict.
    """

    reader_args = [
        {"role": "train", "data_filename": train_path},
        {"role": "validate", "data_filename": val_path},
        {"role": "test", "data_filename": test_path},
    ]

    for reader_arg in reader_args:
        reader_arg["data_file_pattern"] = "{}/*.hdf5".format(
            reader_arg["data_filename"])
        reader_arg["hdf5_key_data"] = "full"
        reader_arg["hdf5_key_responses"] = "unitPar"
        reader_arg["num_responses"] = num_responses
        reader_arg.pop("data_filename")

    readers = []
    for reader_arg in reader_args:
        reader = lbann.reader_pb2.Reader(
            name="hdf5",
            #shuffle=role != "test",
            validation_percent=0,
            absolute_sample_count=0,
            percent_of_data_to_use=1.0,
            disable_labels=True,
            disable_responses=False,
            scaling_factor_int16=1.0,
            **reader_arg)

        readers.append(reader)

    return lbann.reader_pb2.DataReader(reader=readers)

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
    #data_reader.validation_percent = 0.1
    data_reader.python.module = 'dataset3D'
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    return message

if __name__ == '__main__':
    import lbann

    args = construct_lc_launcher_args()
    trainer = lbann.Trainer(args.mini_batch_size)
    model = construct_model(args)
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0002,beta1=0.5,beta2=0.99,eps=1e-8)
    # Runtime parameters/arguments
    environment = lbann.contrib.args.get_distconv_environment(
        num_io_partitions=args.depth_groups)
    if args.dynamically_reclaim_error_signals:
        environment['LBANN_KEEP_ERROR_SIGNALS'] = 0
    else:
        environment['LBANN_KEEP_ERROR_SIGNALS'] = 1
    #lbann_args = ['--use_data_store --num_io_threads=4']
    #lbann_args = ['--num_io_threads=1 --disable_cuda']
    #@todo, parse as args and use data_reader flag to differentiate
    lbann_args = ['--num_io_threads=1']

    # Load data reader from prototext
    data_reader = create_hdf5_data_reader(
        args.train_dir,
        args.val_dir,
        args.test_dir,
        num_responses=args.num_secrets) 

    if(args.use_python_reader):
      print("Using Python Data READER!!!!")
      data_reader = construct_data_reader()

    status = lbann.contrib.launcher.run(trainer,model, data_reader, opt,
                       scheduler=args.scheduler,
                       account=args.account,
                       nodes=args.num_nodes,
                       time_limit=120,
                       #environment=environment,
                       lbann_args=lbann_args,
                       setup_only=False,
                       job_name=args.job_name)
    print(status)
