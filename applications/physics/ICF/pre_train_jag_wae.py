import macc_models
import argparse
from os.path import abspath, dirname, join
import google.protobuf.text_format as txtf
import lbann.contrib.launcher
import lbann.contrib.args

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
lbann.contrib.args.add_scheduler_arguments(parser, 'wae')
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
    '--sample-list-train', action='store', default='/p/vast1/lbann/datasets/JAG/10MJAG/1M_A/index.txt', type=str,
    help='sample list (default index.txt)', metavar='NAME')
parser.add_argument(
    '--sample-list-test', action='store', default='/p/vast1/lbann/datasets/JAG/10MJAG/1M_B/t0_sample_list_multi_10K.txt', type=str,
    help='sample list (default t0_sample_list_multi_10K.txt, 100 samples)', metavar='NAME')
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

if __name__ == '__main__':
    import lbann

    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size,
                            serialize_io=True)
    model = macc_models.construct_jag_wae_model(ydim=args.ydim,
                                                zdim=args.zdim,
                                                mcf=args.mcf,
                                                useCNN=args.useCNN,
                                                dump_models=args.dump_models,
                                                ltfb_batch_interval=args.ltfb_batch_interval,
                                                num_epochs=args.num_epochs)
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
                       time_limit=720,
                       job_name=args.job_name,
                       lbann_args=['--use_data_store --preload_data_store',
                                   f'--metadata={metadata_prototext}',
                                   f'--sample_list_train={args.sample_list_train}',
                                   f'--sample_list_test={args.sample_list_test}',
                                   f'--procs_per_trainer={args.procs_per_trainer}'],
                                   **kwargs)
    print(status)
