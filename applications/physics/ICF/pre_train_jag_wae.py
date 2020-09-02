import macc_models
import argparse
from os.path import abspath, dirname, join
import google.protobuf.text_format as txtf
import lbann.contrib.launcher
import lbann.contrib.args
from lbann.util import str_list

# ==============================================
# Setup and launch experiment
# ==============================================

# Default data reader
model_zoo_dir = dirname(dirname(abspath(__file__)))
data_reader_prototext = join(model_zoo_dir,
                             'data',
                             'jag_conduit_reader.prototext')
metadata_prototext = join(model_zoo_dir,
                             'data',
                             'jag_100M_metadata.prototext')

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='wae', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=128, type=int,
    help='mini-batch size (default: 128)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')
parser.add_argument(
    '--num-nodes', action='store', default=4, type=int,
    help='number of nodes (default: 4)', metavar='NUM')
parser.add_argument(
    '--ppn', action='store', default=4, type=int,
    help='processes per node (default: 4)', metavar='NUM')
parser.add_argument(
    '--ydim', action='store', default=16399, type=int,
    help='image+scalar dim (default: 64*64*4+15=16399)', metavar='NUM')
parser.add_argument(
    '--zdim', action='store', default=20, type=int,
    help='latent space dim (default: 20)', metavar='NUM')
parser.add_argument(
    '--mcf', action='store', default=1, type=int,
    help='model capacity factor (default: 1)', metavar='NUM')
parser.add_argument(
    '--useCNN', action='store', default=False, type=bool,
    help='use CNN', metavar='BOOL')
parser.add_argument(
    '--data-filedir-train', action='store', default='/p/gpfs1/brainusr/datasets/10MJAG/1M_A/', type=str,
    help='data filedir (default train dir is 10MJAG/1M_A)', metavar='NAME')
parser.add_argument(
    '--data-filedir-test', action='store', default='/p/gpfs1/brainusr/datasets/10MJAG/1M_B/', type=str,
    help='data filedir (default test dir is 10MJAG/1M_B)', metavar='NAME')
parser.add_argument(
    '--index-list-train', action='store', default='index.txt', type=str,
    help='index list (default index.txt)', metavar='NAME')
parser.add_argument(
    '--index-list-test', action='store', default='t0_sample_list_multi_10K.txt', type=str,
    help='index list (default t0_sample_list_multi_10K.txt, 100 samples)', metavar='NAME')
parser.add_argument(
    '--dump-outputs', action='store', default='dump_outs', type=str,
    help='dump outputs dir (default: jobdir/dump_outs)', metavar='NAME')
parser.add_argument(
    '--dump-models', action='store', default='dump_models', type=str,
    help='dump models dir (default: jobdir/dump_models)', metavar='NAME')
parser.add_argument(
    '--procs-per-trainer', action='store', default=0, type=int,
    help='processes per trainer (default: 0)', metavar='NUM')
parser.add_argument(
    '--ltfb-batch-interval', action='store', default=0, type=int,
    help='LTFB batch interval (default: 0, no LTFB)', metavar='NUM')
args = parser.parse_args()


def list2str(l):
    return ' '.join(l)

def construct_model():
    """Construct LBANN model.

    JAG Wasserstein autoencoder  model

    """
    import lbann

    # Layer graph
    input = lbann.Input(target_mode='N/A', name='inp_data')
    # data is 64*64*4 images + 15 scalar + 5 param
    #inp_slice = lbann.Slice(input, axis=0, slice_points="0 16399 16404",name='inp_slice')
    inp_slice = lbann.Slice(input, axis=0, slice_points=str_list([0,args.ydim,args.ydim+5]),name='inp_slice')
    gt_y = lbann.Identity(inp_slice,name='gt_y')
    gt_x = lbann.Identity(inp_slice, name='gt_x') #param not used

    zero  = lbann.Constant(value=0.0,num_neurons='1',name='zero')
    one  = lbann.Constant(value=1.0,num_neurons='1',name='one')

    z_dim = 20  #Latent space dim

    z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="20")
    model = macc_models.MACCWAE(args.zdim,args.ydim,cf=args.mcf,use_CNN=args.useCNN)
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
                 lbann.CallbackSaveModel(dir=args.dump_models),
                 lbann.CallbackReplaceWeights(source_layers=list2str(src_layers),
                                      destination_layers=list2str(dst_layers),
                                      batch_interval=2)]

    if(args.ltfb_batch_interval > 0) :
      callbacks.append(lbann.CallbackLTFB(batch_interval=args.ltfb_batch_interval,metric='recon_error',
                                    low_score_wins=True,
                                    exchange_hyperparameters=True))

    # Construct model
    return lbann.Model(args.num_epochs,
                       serialize_io=True,
                       weights=weights,
                       layers=layers,
                       metrics=metrics,
                       objective_function=obj,
                       callbacks=callbacks)


if __name__ == '__main__':
    import lbann

    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size,
                            procs_per_trainer=args.procs_per_trainer)
    model = construct_model()
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0001,beta1=0.9,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader_proto = lbann.lbann_pb2.LbannPB()
    with open(data_reader_prototext, 'r') as f:
      txtf.Merge(f.read(), data_reader_proto)
    data_reader_proto = data_reader_proto.data_reader

    kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
    status = lbann.contrib.launcher.run(trainer,model, data_reader_proto, opt,
                       nodes=args.num_nodes,
                       procs_per_node=args.ppn,
                       time_limit=720,
                       setup_only=True,
                       job_name=args.job_name,
                       lbann_args=['--use_data_store --preload_data_store',
                                   f'--metadata={metadata_prototext}',
                                   f'--index_list_train={args.index_list_train}',
                                   f'--index_list_test={args.index_list_test}',
                                   f'--data_filedir_train={args.data_filedir_train}',
                                   f'--data_filedir_test={args.data_filedir_test}'],
                                   **kwargs)
    print(status)
