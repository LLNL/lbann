#!/usr/tcetmp/bin/python3

import subprocess
import sys
import os
import os.path
import glob
import argparse
import configparser
import datetime

import lbann
import lbann.contrib.launcher

from model import make_model
from data import make_data_reader

gr_ingest_exec = '/usr/WS1/llamag/lbann/applications/graph/communityGAN/havoqgt/build/lassen.llnl.gov/src/ingest_edge_list '
prunejuice_exec = '/usr/WS1/llamag/lbann/applications/graph/communityGAN/havoqgt/build/lassen.llnl.gov/src/run_pattern_matching '

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

  root_dir = os.path.dirname(os.path.realpath(__file__))
  root_dir = os.path.join(root_dir, 'driver')

  config = configparser.ConfigParser()
  config.read(os.path.join(root_dir, 'default.config'))
  config_file = args.config
  if not config_file:
      config_file = os.getenv('COMMUNITY_GAN_CONFIG_FILE')
  if config_file:
      config.read(config_file)

  # Command-line overrides
  if args.graph:
      config.set('Graph', 'graph_file', args.graph)
  if args.graph_store:
      config.set('Graph', 'graph_store', args.graph_store)
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
# Construct a set of motifs in given graph and dump them in a file
# ----------------------------------
def dump_motifs(config):

  # Get the parameters from config
  graph_file = config.get('Graph', 'graph_file', fallback=None)
  graph_store = config.get('Graph', 'graph_store', fallback=None)
  pattern_in_dir = config.get('Pattern', 'pattern_in_dir', fallback=None)
  pattern_out_dir = config.get('Pattern', 'pattern_out_dir', fallback=None)
  match_dir = config.get('Motifs', 'match_dir', fallback=None)
  motifs_out_file = config.get('Motifs', 'motif_file', fallback=None)

  create_motif_file = config.getboolean('Motifs', 'create_motif_file', fallback=None)

  motif_set_file = config.get('Motifs', 'motif_set_file', fallback=None)

  if os.path.exists(match_dir):
    command = '/usr/bin/rm -rf %s/*'%(match_dir)
    os.system(command)
  else:
    command = '/usr/bin/mkdir %s'%(match_dir)
    os.system(command)

  command = '/usr/bin/mv %s/all_ranks_subgraphs/subgraphs_* %s'%(pattern_out_dir, match_dir)
  os.system(command)

  files = os.listdir(match_dir)
  motif_set = set()
  for f in files:
      filename = os.path.join(match_dir, f)
      fd = open(filename, 'r')
      lines = fd.readlines()
      for l in lines:
          tokens = l.split(',')
          amotif = []
          i = 1
          while i < len(tokens)-1:
             amotif.append(int(tokens[i]))
             i = i + 1
          motif_set.add(frozenset(amotif))

      fd.close()

  # Dump all the motifs detected to a file as CSVs
  mfd = open(motifs_out_file, 'w')
  for s in motif_set:
    l = 0
    for v in s:
       mfd.write('%d'%v)
       if l < len(s) - 1:
          mfd.write(', ')

       l = l + 1
    mfd.write('\n')
  mfd.close()

  # List all the motifs found in input graph with thier IDs
  if create_motif_file:
    set_fd = open(motif_set_file, 'w')
    mid = 0
    for s in motif_set:
      set_fd.write('%d: '%mid)
      for v in s:
         set_fd.write('%d '%v)
      set_fd.write('\n')
      mid = mid + 1

    set_fd.close()

# ----------------------------------
# Find all the motifs from given graph and dump them to a file
# ----------------------------------
def find_motifs(config):

  # Get the parameters from config
  graph_file = config.get('Graph', 'graph_file', fallback=None)
  graph_store = config.get('Graph', 'graph_store', fallback=None)
  pattern_in_dir = config.get('Pattern', 'pattern_in_dir', fallback=None)
  pattern_out_dir = config.get('Pattern', 'pattern_out_dir', fallback=None)
  match_dir = config.get('Motifs', 'match_dir', fallback=None)
  motif_file = config.get('Motifs', 'motif_file', fallback=None)

  # NOTE: Following is just a reference to a job script to call prunejuice
  # There are many and better ways to do it.
  script_name = 'motif-find.script'
  f = open(script_name, 'w')
  f.write('#!/usr/bin/csh\n')
  #f.write('#BSUB -G lc\n')
  f.write('#BSUB -nnodes 1\n')
  f.write('#BSUB -W 1:00\n')
  f.write('jsrun -n ALL_HOSTS -r 1 /bin/rm -rf %s'%graph_store + '\n')
  f.write('jsrun -d packed -n ALL_HOSTS -r 1 -a 4 -g 4 -c 40 -b packed:10 -M -gpu %s -p 1 -f 2.00 -o %s %s \n'%(gr_ingest_exec, graph_store, graph_file))
  f.write('jsrun -d packed -n ALL_HOSTS -r 1 -a 4 -g 4 -c 40 -b packed:10 -M -gpu %s -i %s -p %s -o %s\n'%(prunejuice_exec, graph_store, pattern_in_dir, pattern_out_dir))
  f.close()

  # Submit the job and wait for its completion
  cmd = 'bsub -K ' + script_name
  os.system(cmd)

  dump_motifs(config)


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
def do_random_walks (config):

  # Get the parameters from config
  graph_file = config.get('Graph', 'graph_file', fallback=None)
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

  # Prepare rw output directories


  # NOTE: Following is just a reference to a job script to call prunejuice
  # There are many and better ways to do it.
  script_name = 'do-rw.script'
  f = open(script_name, 'w')
  f.write('#!/usr/bin/csh\n')
  #f.write('#BSUB -G lc\n')
  f.write('#BSUB -nnodes 1\n')
  f.write('#BSUB -W 1:00\n')

  # Ingest graph first
  f.write('jsrun -n ALL_HOSTS -r 1 /usr/bin/rm -rf %s\n'%(rw_graph_store))
  f.write('jsrun -d packed -n ALL_HOSTS -r 1 -a 4 -g 4 -c 40 -b packed:10 -M -gpu %s -D -o %s %s \n'%(rw_graph_read_exec, rw_graph_store, graph_file))

  f.write('/usr/bin/rm -rf %s\n'%(rw_walks_store))
  f.write('/usr/bin/mkdir %s\n'%(rw_walks_store))

  f.write('jsrun -d packed -n ALL_HOSTS -r 1 -a 4 -g 4 -c 40 -b packed:10 -M -gpu %s -g %s -o %s/%s -p %s -q %s -l %s -w %s\n'%(rw_exec, rw_graph_store, rw_walks_store, rw_file, p, q, walk_len, num_walkers))

  f.write('/usr/bin/cat %s/%s* > %s\n'%(rw_walks_store, rw_file, rw_out_filename))
  f.write('/usr/bin/rm -rf %s\n'%(rw_walks_store))
  f.close()

  # Submit the job and wait for its completion
  cmd = 'bsub -K ' + script_name
  os.system(cmd)

# ----------------------------------
# Train embeddings
# ----------------------------------

def do_train(
    config_file,
    work_dir,
    motif_size=4,
    walk_length=20,
    num_vertices=1234,
    embed_dim=128,
    learn_rate=1e-2,
    mini_batch_size=512,
    num_epochs=100,
):

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

  # Run LBANN
  lbann.contrib.launcher.run(
    trainer,
    model_,
    data_reader,
    optimizer,
    job_name='lbann_communitygan',
    work_dir=work_dir,
    environment={'LBANN_COMMUNITYGAN_CONFIG_FILE' : config_file},
  )

def main():

  args = add_args()
  config = parse_config(args)
  graph_file = config.get('Graph', 'graph_file', fallback=None)

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
  os.environ['LBANN_NODE2VEC_CONFIG_FILE'] = config_file

  find_motifs(config)
  do_random_walks(config)
  do_train(config_file, args.work_dir)

# find all the motifs from graph
# construct motifs set as CSV
# compute massive random walks from the input graph
# dump random walks to file?

if __name__ == '__main__':
  main()
