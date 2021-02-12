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

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', action='store', default=None, type=str,
        help='configuration file', metavar='FILE')
    parser.add_argument(
        '--work-dir', action='store', default=None, type=str,
        help='working directory', metavar='DIR')
    parser.add_argument(
        '--run', action='store_true',
        help='run directly instead of submitting a batch job')
    return parser.parse_args()

def make_work_dir(args):
    """Make directory to store outputs, logs, and intermediate data.

    If not provided, create a timestamped directory in current working
    directory.

    """
    work_dir = args.work_dir
    if not work_dir:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        work_dir = os.path.join(os.getcwd(), f'{timestamp}_communitygan')
    work_dir = os.path.realpath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir
    return work_dir

def setup_config(args, work_dir):
    """Setup experiment configuration.

    Loads default config and loads user-provided config file. The
    resulting config is written to the work directory.

    """

    # Load default config
    config = configparser.ConfigParser()
    config.read(os.path.join(root_dir, 'default.config'))

    # Read config file if provided
    config_file = args.config
    if not config_file:
        config_file = os.getenv('COMMUNITYGAN_CONFIG_FILE')
    if config_file:
        config.read(config_file)

    # Default parameters for finding motifs
    motif_file = config.get('Motifs', 'file', fallback=None)
    if not motif_file:
        motif_file = os.path.join(work_dir, 'motifs')
    motif_file = os.path.realpath(motif_file)
    config.set('Motifs', 'file', motif_file)
    prunejuice_output_dir = config.get('Motifs', 'prunejuice_output_dir', fallback=None)
    if not prunejuice_output_dir:
        prunejuice_output_dir = os.path.join(work_dir, 'prunejuice_output')
    prunejuice_output_dir = os.path.realpath(prunejuice_output_dir)
    config.set('Motifs', 'prunejuice_output_dir', prunejuice_output_dir)

    # Default parameters for random walks
    walks_file = config.get('Walks', 'file', fallback=None)
    if not walks_file:
        walks_file = os.path.join(work_dir, 'walks')
    walks_file = os.path.realpath(walks_file)
    config.set('Walks', 'file', walks_file)
    distributed_walks_dir = config.get('Walks', 'distributed_walks_dir', fallback=None)
    if not distributed_walks_dir:
        distributed_walks_dir = os.path.join(work_dir, 'distributed_walks')
    distributed_walks_dir = os.path.realpath(distributed_walks_dir)
    config.set('Walks', 'distributed_walks_dir', distributed_walks_dir)

    # Write config file to work directory
    config_file = os.path.join(work_dir, 'experiment.config')
    with open(config_file, 'w') as f:
        config.write(f)
    os.environ['LBANN_COMMUNITYGAN_CONFIG_FILE'] = config_file
    args.config = config_file

    return config, config_file

# ----------------------------------------------------------
# Find motifs
# ----------------------------------------------------------

def setup_motifs(script, config):
    """Add motif finding to batch script."""

    # Get parameters
    graph_file = config.get('Graph', 'file', fallback=None)
    motif_file = config.get('Motifs', 'file', fallback=None)
    pattern_dir = config.get('Motifs', 'pattern_dir', fallback=None)
    graph_ingest_exec = config.get('Motifs', 'graph_ingest_exec', fallback=None)
    distributed_graph_dir = config.get('Motifs', 'distributed_graph_dir', fallback=None)
    prunejuice_exec = config.get('Motifs', 'prunejuice_exec', fallback=None)
    prunejuice_output_dir = config.get('Motifs', 'prunejuice_output_dir', fallback=None)

    # Add motif finding to batch script
    script.add_body_line('')
    script.add_body_line('# Find motifs')
    script.add_parallel_command(['rm', '-rf', distributed_graph_dir], procs_per_node=1)
    script.add_parallel_command([
        graph_ingest_exec,
        '-p 1',
        '-f 2.00',
        f'-o {distributed_graph_dir}',
        graph_file,
    ])
    script.add_command(['rm', '-rf', prunejuice_output_dir])
    script.add_command(['mkdir', '-p', prunejuice_output_dir])
    script.add_command([
        'mkdir',
        '-p',
        os.path.join(prunejuice_output_dir, 'all_ranks_subgraphs'),
    ])
    script.add_parallel_command([
        prunejuice_exec,
        f'-i {distributed_graph_dir}',
        f'-p {pattern_dir}',
        f'-o {prunejuice_output_dir}',
    ])
    script.add_command([
        os.path.realpath(sys.executable),
        os.path.join(root_dir, 'dump_motifs.py'),
        prunejuice_output_dir,
        motif_file,
    ])

# ----------------------------------------------------------
# Perform random walks
# ----------------------------------------------------------

def setup_walks(script, config):
    """Add random walker to batch script."""

    # Get parameters
    graph_file = config.get('Graph', 'file', fallback=None)
    walks_file = config.get('Walks', 'file', fallback=None)
    walk_length = config.getint('Walks', 'walk_length',  fallback=0)
    num_walkers = config.getint('Walks', 'num_walkers', fallback=0)
    p = config.getfloat('Walks', 'p', fallback=-1)
    q = config.getfloat('Walks', 'q', fallback=-1)
    graph_ingest_exec = config.get('Walks', 'graph_ingest_exec', fallback=None)
    distributed_graph_dir = config.get('Walks', 'distributed_graph_dir', fallback=None)
    walk_exec = config.get('Walks', 'walk_exec', fallback=None)
    distributed_walks_dir = config.get('Walks', 'distributed_walks_dir', fallback=None)
    assert (graph_file and walks_file
            and walk_length and num_walkers and p>=0 and q>=0
            and graph_ingest_exec and distributed_graph_dir
            and walk_exec and distributed_walks_dir), \
        'invalid configuration for random walker'

    # Add random walker to batch script
    script.add_body_line('')
    script.add_body_line('# Perform random walks')
    script.add_parallel_command(['rm', '-rf', distributed_graph_dir], procs_per_node=1)
    script.add_parallel_command([
        graph_ingest_exec,
        '-D',
        f'-o {distributed_graph_dir}',
        graph_file,
    ])
    script.add_command(['rm', '-rf', distributed_walks_dir])
    script.add_command(['mkdir', '-p', distributed_walks_dir])
    script.add_parallel_command([
        walk_exec,
        f'-g {distributed_graph_dir}',
        f'-o {os.path.join(distributed_walks_dir, "walks")}',
        f'-p {p}',
        f'-q {q}',
        f'-l {walk_length}',
        f'-w {num_walkers}',
    ])
    script.add_command([
        'cat',
        f'{os.path.join(distributed_walks_dir, "walks")}*',
        '>',
        walks_file,
    ])

# ----------------------------------------------------------
# Train embeddings
# ----------------------------------------------------------

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
    walk_length = config.getint('Walks', 'walk_length', fallback=0)

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

# ----------------------------------------------------------
# Main function
# ----------------------------------------------------------

if __name__ == '__main__':

    # Setup config and work dir
    args = parse_args()
    work_dir = make_work_dir(args)
    config, config_file = setup_config(args, work_dir)

    # Construct batch script
    kwargs = {}
    script = lbann.contrib.launcher.make_batch_script(
        job_name='lbann_communitygan',
        work_dir=work_dir,
        environment={'LBANN_COMMUNITYGAN_CONFIG_FILE' : config_file},
        **kwargs,
    )
    setup_motifs(script, config)
    setup_walks(script, config)
    do_train(script, config)

    # Launch experiment
    if args.run:
        script.run(True)
    else:
        script.submit(True)
