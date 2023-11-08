import Multi_GAN as model
import lbann.contrib.launcher
import lbann.contrib.args
import argparse
import os
from lbann.core.util import get_parallel_strategy_args
import lbann
import lbann.modules as lm

from lbann.contrib.modules.fftshift import FFTShift
from lbann.contrib.modules.radial_profile import RadialProfile
def list2str(l):
    return ' '.join([str(i) for i in l])

#f_transform(x,a=4.0) : return (2*x(x+a))-1

#inv: a * ((1+s)/(1-s))
def f_transform(y, scale=4.0):

    return lbann.Subtract(
                 lbann.SafeDivide(
                       lbann.WeightedSum(y,scaling_factors=2),
                       lbann.Add(lbann.Identity(y),lbann.Constant(value=scale,hint_layer=y))),
                 lbann.Constant(value=1.0, hint_layer=y))

def f_invtransform(y,scale=4.0): ### Transform to original space
    '''
    The inverse of the transformation function that scales the data before training
    '''
    inv_transform = lbann.WeightedSum(
                          lbann.SafeDivide(
                          lbann.Add(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y)),
                          lbann.Subtract(lbann.Constant(value=1.0, hint_layer=y),lbann.Identity(y))),
                          scaling_factors=str(scale))

    return inv_transform


def construct_lc_launcher_args():
    parser = argparse.ArgumentParser()
    lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_cosmo3DGAN')

    # General arguments
    parser.add_argument("--scheduler", type=str, default="lsf")
    parser.add_argument(
        '--mini-batch-size', action='store', default=1, type=int,
        help='mini-batch size (default: 1)', metavar='NUM')
    parser.add_argument(
        '--num-epochs', action='store', default=2, type=int,
        help='number of epochs (default: 2)', metavar='NUM')

    # Model specific arguments
    parser.add_argument(
        '--input-width', action='store', default=64, type=int,
        help='the input spatial width (default: 64)')

    parser.add_argument(
        '--input-channel', action='store', default=1, type=int,
        help='the input channel (default: 1)')

    parser.add_argument(
            '--data-dir', action='store', type=str,
            default  = '/p/vast1/lbann/datasets/exagan/portal.nersc.gov/project/m3363/transfer_data_livermore/64cube_dataset/norm_1_train_val.npy',
            help='dataset directory')

    parser.add_argument(
        '--num-samples', action='store', default=100000, type=int,
        help='the numbe of training/val samples (default: 100000)')

    parser.add_argument(
        '--num-discblocks', action='store', default=1, type=int,
        help='number of discriminator blocks (default: 1)', metavar='NUM')
    # Parallelism arguments
    parser.add_argument(
        '--depth-groups', action='store', type=int, default=2,
        help='the number of processes for the depth dimension (default: 2)')
    parser.add_argument(
        '--dynamically-reclaim-error-signals', action='store_true',
        help='Allow LBANN to reclaim error signals buffers (default: False)')

    parser.add_argument(
        '--use-distconv', action='store_true',
        help='Use distconv')

    parser.add_argument(
        '--compute-mse', action='store_true',
        help='Compute MSE')

    parser.add_argument(
        '--spectral-loss', action='store_true',
        help='Use spectral loss')

    parser.add_argument(
        '--use-bn', action='store_true',
        help='Use batch norm layer')

    parser.add_argument(
        '--dump-outputs', action='store_true',
        help='Dump outputs')

    parser.add_argument(
        '--enable-subgraph', action='store_true',
        help='Enable subgraph parallelism')

    parser.add_argument(
        '--use-hdf5-reader', action='store_true',
        help='Use scalable IO HDF5 reader (default (for now) is Python reader')

    return parser.parse_args()


def construct_model(args):
    """Construct LBANN for conditional CosmoMultiGAN 3D model.

    """
    obj = []
    metrics = []
    callbacks = []

    w  = [args.input_width]*3 
    w.insert(0,args.input_channel)
    _sample_dims = w

    ps = None
    #have model and input ps
    if(args.use_distconv):
      ps = get_parallel_strategy_args(
                sample_groups=args.mini_batch_size,
                height_groups=args.depth_groups)

    g_device = 'GPU'
    if args.use_hdf5_reader:
        input_ = lbann.Input(data_field='img')
        labels = lbann.Input(data_field='sigma')
    else :
        input = lbann.Input(data_field='samples',name='input',device=g_device)
        sl_pt = args.input_width**3
        print('slice_pt', sl_pt) 
        inp_slice = lbann.Slice(input, axis=0, slice_points=[0,sl_pt, sl_pt+1],name='inp_slice')
        input_ = lbann.Reshape(lbann.Identity(inp_slice), dims=_sample_dims,name='in_reshape', device=g_device)
        labels = lbann.Identity(inp_slice,name='xlabel')

    input_ = f_transform(input_) #normalization (see above) here
    x1 = lbann.Identity(input_, parallel_strategy=None, name='x1')
    x2 = lbann.Identity(input_, name='x2') if args.compute_mse or args.spectral_loss else None


    zero  = lbann.Constant(value=0.0,num_neurons=1,name='zero',device=g_device)
    one  = lbann.Constant(value=1.0,num_neurons=1,name='one', device=g_device)

    z = lbann.Reshape(lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims=64, name='noise_vec', device=g_device),
                      dims=[1,64], name='noise_vec_reshape',device=g_device)

    print("RUN ARGS ", args) 

    losses = model.Exa3DMultiGAN(args.input_width,args.input_channel,
                             g_device,ps,use_bn=args.use_bn,num_discblocks=args.num_discblocks,
                             enable_subgraph=args.enable_subgraph)(x1,z,labels)
    print("LEN losses ", len(losses))
 
    layers=list(lbann.traverse_layer_graph([input,z,zero,one]))
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

    block_ids = list(range(0, len(losses)-1, 3))
    for l, i in enumerate(block_ids):
      obj.append(lbann.SigmoidBinaryCrossEntropy([losses[i],one],name='d1_real_bce'+str(i)))
      obj.append(lbann.SigmoidBinaryCrossEntropy([losses[i+1],zero],name='d1_fake_bce'+str(i+1)))
      obj.append(lbann.SigmoidBinaryCrossEntropy([losses[i+2],one],name='d_adv_bce'+str(i+2)))
    
    gen_img = losses[-1] 
    mse = lbann.MeanSquaredError([gen_img, x2], name='MSE') if args.compute_mse else None

    if args.spectral_loss:
      dft_gen_img = lbann.DFTAbs(f_invtransform(gen_img))
      dft_img = lbann.StopGradient(lbann.DFTAbs(f_invtransform(x2)))
         
      ## Adding full spectral loss
      print("SAMPLE DIMS ", _sample_dims)
      gen_fft=FFTShift()(dft_gen_img,_sample_dims)
      gen_spec_prof=RadialProfile()(gen_fft,_sample_dims,63)
        
      img_fft=FFTShift()(dft_img,_sample_dims)
      img_spec_prof=RadialProfile()(img_fft,_sample_dims,63)
      spec_loss = lbann.Log(lbann.MeanSquaredError(gen_spec_prof, img_spec_prof))
      obj.append(lbann.LayerTerm(spec_loss, scale=args.spectral_loss))
      metrics.append(lbann.Metric(spec_loss,name='spec_loss'))


    if (mse is not None):
      obj.append(mse)
      metrics.append(lbann.Metric(mse, name='MSE'))


    callbacks.append(lbann.CallbackPrint())
    callbacks.append(lbann.CallbackTimer())
    callbacks.append(lbann.CallbackStepLearningRate(step=10, amt=0.5))
    callbacks.append(lbann.CallbackGPUMemoryUsage())
    callbacks.append(lbann.CallbackReplaceWeights(source_layers=list2str(src_layers),
                                 destination_layers=list2str(dst_layers),
                                 batch_interval=2))
    if args.dump_outputs:
      callbacks.append(lbann.CallbackDumpOutputs(batch_interval=600,
                       execution_modes='validation',
                       directory='outputs',
                       format='npy',
                       layers=f'{x1.name} {gen_img.name}'))

    # ------------------------------------------
    # Construct model
    # ------------------------------------------

    return lbann.Model(args.num_epochs,
                       weights=weights,
                       layers=layers,
                       objective_function=obj,
                       metrics=metrics,
                       callbacks=callbacks)


if __name__ == '__main__':
    import lbann

    args = construct_lc_launcher_args()
    os.environ['INPUT_WIDTH'] = str(args.input_width)
    os.environ['DATA_DIR'] = args.data_dir
    os.environ['NUM_SAMPLES'] = str(args.num_samples)


    trainer = lbann.Trainer(args.mini_batch_size)
    model = construct_model(args)
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.001,beta1=0.5,beta2=0.99,eps=1e-8)

    # Runtime parameters/arguments
    environment = lbann.contrib.args.get_distconv_environment(
        num_io_partitions=args.depth_groups)

    if args.dynamically_reclaim_error_signals:
        environment['LBANN_KEEP_ERROR_SIGNALS'] = 0
    else:
        environment['LBANN_KEEP_ERROR_SIGNALS'] = 1


    import cgan_data_reader as cdr
    print("Using Python Data READER!!!!")
    data_reader = cdr.construct_python_data_reader()
    if args.use_hdf5_reader:
        # hdf5 data reader
        print("Using HDF5 sample list READER!!!!")
        cur_dir = dirname(abspath(__file__))
        data_reader_prototext = join(cur_dir,
                             'cosmo_hdf5_data',
                             'hdf5_reader.prototext') 
        # Load data reader from prototext
        data_reader_proto = lbann.lbann_pb2.LbannPB()
        with open(data_reader_prototext, 'r') as f:
            txtf.Merge(f.read(), data_reader_proto)
        data_reader = data_reader_proto.data_reader

    
    #Remove cosmoflow/hdf5 stuff
    environment.pop('LBANN_DISTCONV_COSMOFLOW_PARALLEL_IO')
    environment.pop('LBANN_DISTCONV_NUM_IO_PARTITIONS')
    lbann_args = ['--num_io_threads=1']

    environment['INPUT_WIDTH'] = str(args.input_width)
    environment['DATA_DIR'] = args.data_dir
    environment['NUM_SAMPLES'] = str(args.num_samples)

    print('LBANN args ', lbann_args)
    print("LBANN ENV VAR ", environment)

    status = lbann.contrib.launcher.run(trainer,model, data_reader, opt,
                       scheduler=args.scheduler,
                       account='exalearn',
                       partition='pbatch',
                       nodes=args.nodes,
                       procs_per_node=args.procs_per_node,
                       time_limit=720,
                       environment=environment,
                       lbann_args=lbann_args,
                       setup_only=False,
                       batch_job=False,
                       job_name=args.job_name)
    print(status)
