import macc_models
import argparse
import os
from os.path import abspath, dirname, join
import google.protobuf.text_format as txtf
import lbann.contrib.launcher
import lbann.contrib.args
from lbann.util import str_list
import datetime

# ==============================================
# Setup and launch experiment
# ==============================================

# Default data reader
cur_dir = dirname(abspath(__file__))
data_reader_prototext = join(dirname(cur_dir),
                             'data',
                             'eval_jag_conduit_lassen.prototext')
metadata_prototext = join(dirname(cur_dir),
                             'data',
                             'jag_100M_metadata.prototext')

# Initialize LBANN inf executable
lbann_exe = abspath(lbann.lbann_exe())
lbann_exe = join(dirname(lbann_exe), 'lbann_inf')

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='eval', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=4096, type=int,
    help='mini-batch size (default: 128)', metavar='NUM')
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
    '--xdim', action='store', default=5, type=int,
    help='input (x) dim (default: 5)', metavar='NUM')
parser.add_argument(
    '--wae_mcf', action='store', default=1, type=int,
    help='wae model capacity factor (default: 1)', metavar='NUM')
parser.add_argument(
    '--surrogate_mcf', action='store', default=1, type=int,
    help='surrogate model capacity factor (default: 1)', metavar='NUM')
parser.add_argument(
    '--lamda-cyc', action='store', default=1e-3, type=float,
    help='lamda-cyc (default: 1e-3)', metavar='NUM')
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
    '--index-list-train', action='store', default='index_eight.txt', type=str,
    help='index list (default index_eight 8 samples)', metavar='NAME')
parser.add_argument(
    '--index-list-test', action='store', default='t2_index.txt', type=str,
    help='index list (default index.txt)', metavar='NAME')
parser.add_argument(
    '--percent-of-data-to-use', action='store', default=0.01, type=float,
    help='percent of data to use (default: 0.01)', metavar='NUM')
parser.add_argument(
    '--dump-outputs', action='store', default='dump_outs', type=str,
    help='dump outputs dir (default: jobdir/dump_outs)', metavar='NAME')
parser.add_argument(
    '--pretrained-dir', action='store', default=None, type=str,
    help='pretrained WAE surrogate dir  (default: ' ')', metavar='NAME')
parser.add_argument(
    '--procs-per-trainer', action='store', default=0, type=int,
    help='processes per trainer (default: 0)', metavar='NUM')
args = parser.parse_args()

print("Pretrained dir ", args.pretrained_dir)
assert args.pretrained_dir, "evaluate script asssumes a pretrained MaCC model"

def list2str(l):
    return ' '.join(l)

def construct_model():
    """Construct MACC surrogate model.

    See https://arxiv.org/pdf/1912.08113.pdf model architecture and other details

    """
    import lbann

    # Layer graph
    input = lbann.Input(target_mode='N/A',name='inp_data')
    # data is 64*64*4 images + 15 scalar + 5 param
    inp_slice = lbann.Slice(input, axis=0, slice_points=str_list([0,args.ydim,args.ydim+args.xdim]),name='inp_slice')
    gt_y = lbann.Identity(inp_slice,name='gt_y')
    gt_x = lbann.Identity(inp_slice, name='gt_x') #param not used

    zero  = lbann.Constant(value=0.0,num_neurons='1',name='zero')
    one  = lbann.Constant(value=1.0,num_neurons='1',name='one')


    z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="20")
    wae = macc_models.MACCWAE(args.zdim,args.ydim,cf=args.wae_mcf,use_CNN=args.useCNN) #pretrained, freeze
    inv = macc_models.MACCInverse(args.xdim,cf=args.surrogate_mcf)
    fwd = macc_models.MACCForward(args.zdim,cf=args.surrogate_mcf)


    y_pred_fwd = wae.encoder(gt_y)

    param_pred_ = wae.encoder(gt_y)
    input_fake = inv(param_pred_)

    output_cyc = fwd(input_fake)
    y_image_re2  = wae.decoder(output_cyc)

    '''**** Train cycleGAN input params <--> latent space of (images, scalars) ****'''
    output_fake = fwd(gt_x)
    y_image_re = wae.decoder(output_fake)

    y_out = wae.decoder(y_pred_fwd)

    param_pred2_ = wae.encoder(y_image_re)
    input_cyc = inv(param_pred2_)

    L_l2_x =  lbann.MeanSquaredError(input_fake,gt_x) #(x,inv(enc(y)), (encoder+)inverse loss
    L_cyc_x = lbann.MeanSquaredError(input_cyc,gt_x)  #param, x cycle loss, from latent space

    L_l2_y =  lbann.MeanSquaredError(output_fake,y_pred_fwd) #pred error into latent space (enc(y),fw(x))
    L_cyc_y = lbann.MeanSquaredError(output_cyc,y_pred_fwd) # pred error into latent space (enc(y), fw(inv(enc(y))))


    #@todo slice here to separate scalar from image
    img_sca_loss = lbann.MeanSquaredError(y_image_re,gt_y) # (y,dec(fw(x))) #forward model to decoder, no latent space
    dec_fw_inv_enc_y = lbann.MeanSquaredError(y_image_re2,gt_y) #(y, dec(fw(inv(enc(y))))) y->enc_z->x'->fw_z->y'
    wae_loss  = lbann.MeanSquaredError(y_out,gt_y) #(y, dec(enc(y)) '
    #L_cyc = L_cyc_y + L_cyc_x
    L_cyc = lbann.Add(L_cyc_y, L_cyc_x)

    #loss_gen0  = L_l2_y + lamda_cyc*L_cyc
    loss_gen0  = lbann.WeightedSum([L_l2_y,L_cyc], scaling_factors=f'1 {args.lamda_cyc}')
    loss_gen1  = lbann.WeightedSum([L_l2_x,L_cyc_y], scaling_factors=f'1 {args.lamda_cyc}')
    #loss_gen1  =  L_l2_x + lamda_cyc*L_cyc_y


    conc_out = lbann.Concatenation([gt_x,wae_loss,img_sca_loss,dec_fw_inv_enc_y,
                                    L_l2_x], name='x_errors')
    layers = list(lbann.traverse_layer_graph(input))
    weights = set()
    for l in layers:
      weights.update(l.weights)

    # Setup objective function
    obj = lbann.ObjectiveFunction([loss_gen0,loss_gen1])
    # Initialize check metric callback
    metrics = [lbann.Metric(img_sca_loss, name='img_re1'),
               lbann.Metric(dec_fw_inv_enc_y, name='img_re2'),
               lbann.Metric(wae_loss, name='wae_loss'),
               lbann.Metric(L_l2_x, name='inverse loss'),
               lbann.Metric(L_cyc_y, name='output cycle loss'),
               lbann.Metric(L_cyc_x, name='param cycle loss')]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackDumpOutputs(layers=f'{conc_out.name}',
                                           execution_modes='test',
                                           directory=args.dump_outputs,
                                           batch_interval=1,
                                           format='npy'),
                 lbann.CallbackTimer()]

    # Construct model
    num_epochs =1
    return lbann.Model(num_epochs,
                       weights=weights,
                       layers=layers,
                       serialize_io=True,
                       metrics=metrics,
                       objective_function=obj,
                       callbacks=callbacks)


if __name__ == '__main__':

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
                       lbann_exe,
                       scheduler='lsf',
                       partition='pdebug',
                       nodes=args.num_nodes,
                       procs_per_node=args.ppn,
                       time_limit=30,
                       setup_only=False,
                       batch_job=False,
                       job_name=args.job_name,
                       lbann_args=['--preload_data_store --use_data_store --load_model_weights_dir_is_complete',
                                   f'--metadata={metadata_prototext}',
                                   f'--load_model_weights_dir={args.pretrained_dir}',
                                   f'--index_list_test={args.index_list_test}',
                                   f'--data_filedir_test={args.data_filedir_test}'],
                                   **kwargs)
    print(status)
