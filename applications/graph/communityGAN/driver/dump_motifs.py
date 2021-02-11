import configparser
import os

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

if __name__ == '__main__':

    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', action='store', default=None, type=str,
        help='configuration file'
    )
    args = parser.parse_args()

    # Parse config file
    config = configparser.ConfigParser()
    config.read(os.path.join(args.config))

    # Convert motifs to format readable by LBANN
    dump_motifs(config)
