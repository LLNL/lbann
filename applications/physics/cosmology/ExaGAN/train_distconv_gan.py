import DistConvGAN as model
import lbann.contrib.launcher
import lbann.contrib.args
import argparse
import os
from lbann.core.util import get_parallel_strategy_args
import lbann
import lbann.modules as lm

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

    # Parallelism arguments
    parser.add_argument(
        '--depth-groups', action='store', type=int, default=2,
        help='the number of processes for the depth dimension (default: 2)')
    parser.add_argument(
        '--height-groups', action='store', type=int, default=1,
        help='the number of processes for the height dimension (default: 1)')
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
        '--use-bn', action='store_true',
        help='Use batch norm layer')

    return parser.parse_args()


def construct_model(args):
    """Construct LBANN for CosmoGAN 3D model.

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
          depth_groups=args.depth_groups,
          height_groups=args.height_groups,
      )

    g_device = 'GPU'
    input_ = lbann.Input(name='input', data_field='samples')
    input_ = lbann.Reshape(input_, dims=_sample_dims,name='in_reshape', device=g_device),
    x1 = lbann.Identity(input_, parallel_strategy=None, name='x1')
    x2 = lbann.Identity(input_, name='x2') if args.compute_mse else None

    zero  = lbann.Constant(value=0.0,num_neurons=[1],name='zero',device=g_device)
    one  = lbann.Constant(value=1.0,num_neurons=[1],name='one', device=g_device)

    z = lbann.Reshape(lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims=[64], name='noise_vec', device=g_device),
                      dims=[1, 64], name='noise_vec_reshape',device=g_device)
    print("RUN ARGS ", args)

    d1_real,d1_fake,d_adv, gen_img = model.Exa3DGAN(args.input_width,args.input_channel,
                             g_device,ps,use_bn=args.use_bn)(x1,z)

    layers=list(lbann.traverse_layer_graph([d1_real, d1_fake]))
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

    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,one],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zero],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,one],name='d_adv_bce')
    mse = lbann.MeanSquaredError([gen_img, x2], name='MSE') if args.compute_mse else None

    obj.append(d1_real_bce)
    obj.append(d1_fake_bce)
    obj.append(d_adv_bce)

    metrics.append(lbann.Metric(d_adv_bce, name='d_adv_bce'))
    metrics.append(lbann.Metric(d1_real_bce, name='d1_real_bce'))
    metrics.append(lbann.Metric(d1_fake_bce, name='d1_fake_bce'))
    if (mse is not None):
      obj.append(mse)
      metrics.append(lbann.Metric(mse, name='MSE'))

    callbacks.append(lbann.CallbackPrint())
    callbacks.append(lbann.CallbackTimer())
    callbacks.append(lbann.CallbackGPUMemoryUsage())

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

    trainer = lbann.Trainer(args.mini_batch_size)
    model = construct_model(args)
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0002,beta1=0.5,beta2=0.99,eps=1e-8)

    # Runtime parameters/arguments
    environment = lbann.contrib.args.get_distconv_environment(
        num_io_partitions=args.depth_groups)
    environment['INPUT_WIDTH'] = os.environ['INPUT_WIDTH']
    environment['DATA_DIR'] = os.environ['DATA_DIR']
    environment['LBANN_DISTCONV_HALO_EXCHANGE'] = 'AL'
    environment['LBANN_DISTCONV_TENSOR_SHUFFLER'] = 'AL'

    if args.dynamically_reclaim_error_signals:
        environment['LBANN_KEEP_ERROR_SIGNALS'] = 0
    else:
        environment['LBANN_KEEP_ERROR_SIGNALS'] = 1


    import construct_data_reader as cdr
    print("Using Python Data READER!!!!")
    data_reader = cdr.construct_python_data_reader()
    #Remove cosmoflow/hdf5 stuff
    environment.pop('LBANN_DISTCONV_COSMOFLOW_PARALLEL_IO')
    environment.pop('LBANN_DISTCONV_NUM_IO_PARTITIONS')
    lbann_args = ['--num_io_threads=1']

    print('LBANN args ', lbann_args)
    print("LBANN ENV VAR ", environment)

    status = lbann.contrib.launcher.run(trainer,model, data_reader, opt,
                       scheduler=args.scheduler,
                       account='exalearn',
                       partition='pbatch',
                       nodes=args.nodes,
                       procs_per_node=args.procs_per_node,
                       time_limit=480,
                       environment=environment,
                       lbann_args=lbann_args,
                       setup_only=False,
                       batch_job=False,
                       job_name=args.job_name)
    print(status)
