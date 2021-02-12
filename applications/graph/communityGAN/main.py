import argparse
import configparser
import datetime
import os
import subprocess
import sys

import lbann
import lbann.contrib.launcher
import numpy as np

from data import make_data_reader
from model import make_model

root_dir = os.path.dirname(os.path.realpath(__file__))

# ----------------------------------
# Add/Define arguments to command-line parser
# ----------------------------------
def add_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', action='store', default=None, type=str,
        help='data config file', metavar='FILE')
    parser.add_argument(
        '--work-dir', action='store', default=None, type=str,
        help='working directory', metavar='DIR')
    parser.add_argument(
        '--run', action='store_true',
        help='run directly instead of submitting a batch job')
    parser.add_argument(
        '--graph', action='store', default=None, type=str,
        help='edgelist file',
        metavar='FILE')
    parser.add_argument(
        '--graph_store', action='store', default=None, type=str,
        help='where to store input graph',
        metavar='FILE')
    parser.add_argument(
        '--pattern_in_dir', action='store', default=None, type=str,
        help='input motifs directory',
        metavar='FILE')
    parser.add_argument(
        '--pattern_out_dir', action='store', default=None, type=str,
        help='where to store match results',
        metavar='FILE')
    parser.add_argument(
        '--match_dir', action='store', default=None, type=str,
        help='directory to store match results',
        metavar='FILE')
    parser.add_argument(
        '--motif_file', action='store', default=None, type=str,
        help='name of motif CSV file',
        metavar='FILE')
    parser.add_argument(
        '--create_motif_file', action='store', default=False, type=bool,
        help='option to create motif set file',
        metavar='FILE')
    parser.add_argument(
        '--motif_set_file', action='store', default=None, type=str,
        help='the name of file to store motifs with their IDs (when create_motif_file flag is set)',
        metavar='FILE')
    parser.add_argument(
        '--rw_walks_store', action='store', default=None, type=str,
        help='place to park rw paths',
        metavar='FILE')
    parser.add_argument(
        '--rw_out_filename', action='store', default=None, type=str,
        help='base name for RW results',
        metavar='FILE')
    parser.add_argument(
        '--rw_walk_len', action='store', default=None, type=str,
        help='the length of each walk',
        metavar='FILE')
    parser.add_argument(
        '--rw_num_walkers', action='store', default=None, type=str,
        help='the number of walkers per each vertex',
        metavar='FILE')
    parser.add_argument(
        '--rw_p', action='store', default=None, type=str,
        help='the p value for node2vec rw',
        metavar='FILE')
    parser.add_argument(
        '--rw_q', action='store', default=None, type=str,
        help='the q value for node2vec rw',
        metavar='FILE')

    args = parser.parse_args()
    return args


# ----------------------------------
# Parse config file
# ----------------------------------
def parse_config(args):

    config = configparser.ConfigParser()
    config.read(os.path.join(root_dir, 'default.config'))
    config_file = args.config
    if not config_file:
        config_file = os.getenv('COMMUNITYGAN_CONFIG_FILE')
    if config_file:
        config.read(config_file)

    # Command-line overrides
    if args.graph:
        config.set('Graph', 'file', args.graph)
    if args.graph_store:
        config.set('Graph', 'ingest_dir', args.graph_store)
    if args.pattern_in_dir:
        config.set('Pattern', 'pattern_in_dir', args.pattern_in_dir)
    if args.pattern_out_dir:
        config.set('Pattern', 'pattern_out_dir', args.pattern_out_dir)
    if args.match_dir:
        config.set('Motifs', 'match_dir', args.match_dir)
    if args.motif_file:
        config.set('Motifs', 'motif_file', args.motif_file)
    if args.create_motif_file:
        config.set('Motifs', 'create_motif_file', args.create_motif_file)
    if args.motif_set_file:
        config.set('Motifs', 'motif_set_file', args.motif_set_file)
    if args.rw_walks_store:
        config.set('RW', 'rw_walks_store', args.rw_walks_store)
    if args.rw_out_filename:
        config.set('RW', 'rw_out_filename', args.rw_out_filename)
    if args.rw_walk_len:
        config.set('RW', 'rw_walk_len', args.rw_walk_len)
    if args.rw_num_walkers:
        config.set('RW', 'rw_num_walkers', args.rw_num_walkers)
    if args.rw_p:
        config.set('RW', 'rw_p', args.rw_p)
    if args.rw_q:
        config.set('RW', 'rw_q', args.rw_q)

    return config

# ----------------------------------
# Find all the motifs from given graph and dump them to a file
# ----------------------------------

def find_motifs(script, config, config_file):

    # Get the parameters from config
    graph_file = config.get('Graph', 'file', fallback=None)
    graph_store = config.get('Graph', 'ingest_dir', fallback=None)
    pattern_in_dir = config.get('Pattern', 'pattern_in_dir', fallback=None)
    pattern_out_dir = config.get('Pattern', 'pattern_out_dir', fallback=None)
    match_dir = config.get('Motifs', 'match_dir', fallback=None)
    motif_file = config.get('Motifs', 'motif_file', fallback=None)

    gr_ingest_exec = '/usr/WS1/llamag/lbann/applications/graph/communityGAN/havoqgt/build/lassen.llnl.gov/src/ingest_edge_list '
    prunejuice_exec = '/usr/WS1/llamag/lbann/applications/graph/communityGAN/havoqgt/build/lassen.llnl.gov/src/run_pattern_matching '

    # NOTE: Following is just a reference to a job script to call prunejuice
    # There are many and better ways to do it.
    script.add_body_line('')
    script.add_body_line('# Find motifs')
    script.add_parallel_command(['rm', '-rf', graph_store], procs_per_node=1)
    script.add_parallel_command([
        gr_ingest_exec,
        '-p 1',
        '-f 2.00',
        f'-o {graph_store}',
        graph_file,
    ])
    script.add_parallel_command([
        prunejuice_exec,
        f'-i {graph_store}',
        f'-p {pattern_in_dir}',
        f'-o {pattern_out_dir}',
    ])
    script.add_command([
        os.path.realpath(sys.executable),
        os.path.join(root_dir, 'dump_motifs.py'),
        config_file,
    ])

'''
# Compute initial values for Theta_G and Theta_D, via graph embedding and initialize the matrices

# Spawn Keita's RW code in background to generate RW paths for all the vertices in graph
# Need to touch, Keita's side code to repeatedly generate random walks as long as the
# output RW files are consumed and removed
rw = subprocess.Popen(["path-to-keita's", "parameters ..."])

# Start LBANN process here
# CommunityGAN produces SW files and LBANN consumes/erase these files
# Start LBANN learing
# End of LBANN learning

# Stop RW gen code
rw.terminate()
'''

# ----------------------------------
# Generate random walks from given input graph
# ----------------------------------

def do_random_walks(script, config):

    # Get the parameters from config
    graph_file = config.get('Graph', 'file', fallback=None)
    rw_graph_store = '/dev/shm/gr/'
    rw_walks_store = config.get('RW', 'rw_walks_store', fallback=None)
    rw_file = 'rw_outs'
    rw_out_filename = config.get('RW', 'rw_out_filename', fallback=None)

    walk_len = config.get('RW', 'walk_len',  fallback=80)
    num_walkers = config.get('RW', 'num_walkers', fallback=40)
    p = config.get('RW', 'p', fallback=1.0)
    q = config.get('RW', 'q', fallback=1.0)

    rw_graph_read_exec = '/usr/workspace/llamag/lbann/applications/graph/communityGAN/largescale_node2vec/build/lassen.llnl.gov/src/construct_dist_graph '
    rw_exec = '/usr/workspace/llamag/lbann/applications/graph/communityGAN/largescale_node2vec/build/lassen.llnl.gov/src/run_dist_node2vec_rw '

    script.add_body_line('')
    script.add_body_line('# Perform random walks')

    # Ingest graph
    script.add_parallel_command(['rm', '-rf', rw_graph_store], procs_per_node=1)
    script.add_parallel_command([
        rw_graph_read_exec,
        '-D',
        f'-o {rw_graph_store}',
        graph_file,
    ])

    # Perform random walks
    script.add_command(['rm', '-rf', rw_walks_store])
    script.add_command(['mkdir', rw_walks_store])
    script.add_parallel_command([
        rw_exec,
        f'-g {rw_graph_store}',
        f'-o {os.path.join(rw_walks_store, rw_file)}',
        f'-p {p}',
        f'-q {q}',
        f'-l {walk_len}',
        f'-w {num_walkers}',
    ])
    script.add_command([
        'cat',
        f'{os.path.join(rw_walks_store, rw_file)}*',
        '>',
        rw_out_filename,
    ])
    script.add_command(['rm', '-rf', rw_walks_store])

# ----------------------------------
# Train embeddings
# ----------------------------------

def do_train(
    script,
    config,
    motif_size=4,
    embed_dim=128,
    learn_rate=1e-2,
    mini_batch_size=512,
    num_epochs=100,
):

    # Get parameters from config file
    num_vertices = config.getint('Graph', 'num_vertices', fallback=0)
    walk_length = config.getint('RW', 'rw_walk_len', fallback=0)

    # Determine number of vertices, if needed
    if not num_vertices:
        graph_file = config.get('Graph', 'file', fallback=None)
        assert graph_file, 'Graph file not provided'
        num_vertices = np.loadtxt(graph_file, dtype=int).max() + 1

    # Construct LBANN objects
    trainer = lbann.Trainer(
        mini_batch_size=mini_batch_size,
        num_parallel_readers=0,
    )
    model_ = make_model(
        motif_size,
        walk_length,
        num_vertices,
        embed_dim,
        learn_rate,
        num_epochs,
    )
    optimizer = lbann.SGD(learn_rate=learn_rate)
    data_reader = make_data_reader()

    # Add LBANN invocation to batch script
    prototext_file = os.path.join(script.work_dir, 'experiment.prototext')
    lbann.proto.save_prototext(
        prototext_file,
        trainer=trainer,
        model=model_,
        data_reader=data_reader,
        optimizer=optimizer,
    )
    script.add_body_line('')
    script.add_body_line('# Train embeddings')
    script.add_parallel_command([
        lbann.lbann_exe(),
        f'--prototext={prototext_file}',
        f'--num_io_threads=1',
    ])

# find all the motifs from graph
# construct motifs set as CSV
# compute massive random walks from the input graph
# dump random walks to file?

if __name__ == '__main__':
    args = add_args()
    config = parse_config(args)

    # Create work directory
    # Note: Default is timestamped directory in cwd
    if not args.work_dir:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.work_dir = os.path.join(os.getcwd(), f'{timestamp}_communitygan')
    args.work_dir = os.path.realpath(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)

    # Write config file to work directory
    config_file = os.path.join(args.work_dir, 'experiment.config')
    with open(config_file, 'w') as f:
        config.write(f)
    os.environ['LBANN_COMMUNITYGAN_CONFIG_FILE'] = config_file

    # Construct batch script
    kwargs = {}
    script = lbann.contrib.launcher.make_batch_script(
        job_name='lbann_communitygan',
        work_dir=args.work_dir,
        environment={'LBANN_COMMUNITYGAN_CONFIG_FILE' : config_file},
        **kwargs,
    )
    find_motifs(script, config, config_file)
    do_random_walks(script, config)
    do_train(script, config)

    # Run LBANN
    if args.run:
        script.run(True)
    else:
        script.submit(True)
