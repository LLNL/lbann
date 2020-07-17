"""Learn embedding weights with LBANN."""
import argparse
import os.path
import numpy as np

import lbann
import lbann.contrib.launcher
import lbann.contrib.args

import data.data_readers
import utils
import utils.graph
import utils.snap

root_dir = os.path.dirname(os.path.realpath(__file__))

# ----------------------------------
# Options
# ----------------------------------

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_node2vec', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--graph', action='store', default='youtube', type=str,
    help='graph name (see utils.snap.download_graph) or edgelist file',
    metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-iterations', action='store', default=1000, type=int,
    help='number of epochs (default: 1000)', metavar='NUM')
parser.add_argument(
    '--latent-dim', action='store', default=128, type=int,
    help='latent space dimensions (default: 128)', metavar='NUM')
parser.add_argument(
    '--learning-rate', action='store', default=-1, type=float,
    help='learning rate (default: 0.25*mbsize)', metavar='VAL')
parser.add_argument(
    '--work-dir', action='store', default=None, type=str,
    help='working directory', metavar='DIR')
parser.add_argument(
    '--offline-walks', action='store_true',
    help='perform random walks offline')
parser.add_argument(
    '--disable-dist-embeddings', action='store_true',
    help='disable distributed embedding layers')
args = parser.parse_args()

# Default learning rate
# Note: Learning rate in original word2vec is 0.025
if args.learning_rate < 0:
    args.learning_rate = 0.25 * args.mini_batch_size

# Random walk options
epoch_size = 100 * args.mini_batch_size
walk_length = 500
return_param = 0.25
inout_param = 0.25
num_negative_samples = 200

# ----------------------------------
# Create data reader
# ----------------------------------

# Download graph if needed
if os.path.exists(args.graph):
    graph_file = args.graph
else:
    graph_file = utils.snap.download_graph(args.graph)

# Construct data reader
if args.offline_walks:
    # Note: Graph and walk parameters are fully specified in module
    # for offline walks
    import data.offline_walks
    graph_file = data.offline_walks.graph_file
    epoch_size = data.offline_walks.num_samples()
    walk_length = data.offline_walks.walk_length
    return_param = data.offline_walks.return_param
    inout_param = data.offline_walks.inout_param
    num_negative_samples = data.offline_walks.num_negative_samples
    reader = data.data_readers.make_offline_data_reader()
else:
    # Note: Preprocess graph with HavoqGT and store in shared memory
    # before starting LBANN.
    distributed_graph_file = '/dev/shm/graph'
    reader = data.data_readers.make_online_data_reader(
        graph_file=distributed_graph_file,
        epoch_size=epoch_size,
        walk_length=walk_length,
        return_param=return_param,
        inout_param=inout_param,
        num_negative_samples=num_negative_samples,
    )

# Parse graph file to get number of vertices
num_vertices = utils.graph.max_vertex_index(graph_file) + 1

# ----------------------------------
# Embedding weights
# ----------------------------------

embeddings_weights = lbann.Weights(
    initializer=lbann.NormalInitializer(
        mean=0, standard_deviation=1/args.latent_dim,
    ),
    name='embeddings',
)

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Embedding vectors, including negative sampling
# Note: Input is sequence of vertex IDs
input_ = lbann.Input()
if args.disable_dist_embeddings:
    embeddings = lbann.Embedding(
        input_,
        weights=embeddings_weights,
        num_embeddings=num_vertices,
        embedding_dim=args.latent_dim,
    )
else:
    embeddings = lbann.DistEmbedding(
        input_,
        weights=embeddings_weights,
        num_embeddings=num_vertices,
        embedding_dim=args.latent_dim,
        sparse_sgd=True,
        learning_rate=args.learning_rate,
    )
num_encoder_embeddings = walk_length - 1
num_decoder_embeddings = num_negative_samples + walk_length - 1
encoder_embeddings = lbann.Slice(
    embeddings,
    axis=0,
    slice_points=f'{num_negative_samples+1} {num_negative_samples+walk_length}',
)
decoder_embeddings = lbann.Slice(
    embeddings,
    axis=0,
    slice_points=f'0 {num_negative_samples+walk_length-1}',
)

# Multiply encoder and decoder embeddings
preds = lbann.MatMul(decoder_embeddings, encoder_embeddings, transpose_b=True)
preds_slice = lbann.Slice(
    preds,
    axis=0,
    slice_points=f'0 {num_negative_samples} {num_negative_samples+walk_length-1}',
)
preds_negative = lbann.Identity(preds_slice)
preds_positive = lbann.Identity(preds_slice)

# Loss function for positive samples
mask_dims = (num_decoder_embeddings-num_negative_samples, num_encoder_embeddings)
mask_decay = 0.9
mask = np.zeros((walk_length-1,walk_length-1))
for i in range(walk_length-1):
    for j in range(i, walk_length-1):
        mask[i,j] = (1-mask_decay) * mask_decay**(j-i)
mask = lbann.Weights(
    initializer=lbann.ValueInitializer(values=utils.str_list(np.nditer(mask))),
    optimizer=lbann.NoOptimizer(),
)
mask = lbann.WeightsLayer(dims=utils.str_list(mask_dims), weights=mask)
obj_positive = lbann.LogSigmoid(preds_positive)
obj_positive = lbann.MatMul(
    lbann.Reshape(obj_positive, dims='1 -1'),
    lbann.Reshape(mask, dims='1 -1'),
    transpose_b=True,
)

# Loss function for negative samples
obj_negative = lbann.WeightedSum(preds_negative, scaling_factors='-1')
obj_negative = lbann.LogSigmoid(obj_negative)
obj_negative = lbann.Reduction(obj_negative, mode='average')

# Loss function
obj = [
    lbann.LayerTerm(obj_positive, scale=-1/(walk_length-1)),
    lbann.LayerTerm(obj_negative, scale=-1),
]

# ----------------------------------
# Run LBANN
# ----------------------------------

# Create optimizer
opt = lbann.SGD(learn_rate=args.learning_rate)

# Create LBANN objects
iterations_per_epoch = utils.ceildiv(epoch_size, args.mini_batch_size)
num_epochs = utils.ceildiv(args.num_iterations, iterations_per_epoch)
trainer = lbann.Trainer(
    mini_batch_size=args.mini_batch_size,
    num_parallel_readers=0,
)
callbacks = [
    lbann.CallbackPrint(),
    lbann.CallbackTimer(),
    lbann.CallbackDumpWeights(directory='embeddings',
                              epoch_interval=num_epochs),
    lbann.CallbackPrintModelDescription(),
]
model = lbann.Model(
    num_epochs,
    layers=lbann.traverse_layer_graph(input_),
    objective_function=obj,
    callbacks=callbacks,
)

# Create batch script
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
script = lbann.contrib.launcher.make_batch_script(
    job_name=args.job_name,
    work_dir=args.work_dir,
    **kwargs,
)

# Preprocess graph data with HavoqGT if needed
if not args.offline_walks:
    ingest_graph_exe = os.path.join(
        root_dir,
        'build',
        'havoqgt',
        'src',
        'ingest_edge_list',
    )
    script.add_parallel_command([
        ingest_graph_exe,
        f'-o {distributed_graph_file}',
        f'-d {2**30}',
        '-u 1',
        graph_file,
    ])

# LBANN invocation
prototext_file = os.path.join(script.work_dir, 'experiment.prototext')
lbann.proto.save_prototext(
    prototext_file,
    trainer=trainer,
    model=model,
    data_reader=reader,
    optimizer=opt,
)
script.add_parallel_command([
    lbann.lbann_exe(),
    f'--prototext={prototext_file}',
    f'--num_io_threads=1',
])

# Run LBANN
script.run()
