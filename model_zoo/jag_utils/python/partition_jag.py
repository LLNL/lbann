#!/usr/bin/env python

'''
=====================================================================
This code partitions the 100M JAG set as follows:

from 100M:
  set A = 10M random samples
  set B = remaining (90M)

from set B:
  for j in range 0..9 :
    extract 256 sets (C_j_0, C_j_1, ..., c_j_255)

Note: each C_j_i has ~90M / 256 samples
Note: also permute the lines in the sample lists for the C sets
=====================================================================
'''

# Change these as appropriate
output_dir = '/p/gpfs1/hysom/jag_sample_lists_2'
exe = '/usr/workspace/wsb/hysom/lbann/build/gnu.Release.lassen.llnl.gov/lbann/build/model_zoo/jag_utils/select_samples'

# Set to false for non-repeatable output
repeat_time = True

num_multi_lists = 256
validation_sample_count = 10000000
data_sub_dir = '100M/'
num_trials = 10

# uncomment the following for initial testing
'''
num_multi_lists = 5 
validation_sample_count = 100000
data_sub_dir = '10MJAG/1M_A/'
num_trials = 2
'''

#Note: print statements from this script begin with 'PY: '
#      to distinguish them from print statements fromt
#      the c++ executable "select_samples"

#============================================================
# All that follows should be OK for pascal or lassen
#============================================================

# Set data directory for pascal or lassen
import socket
host = socket.gethostname()
data_dir = '/p/lustre2/'
if host.find('lassen') != -1 or host.find('sierra') != -1 :
  data_dir = '/p/gpfs1/'

# Remaining global variables
jag_dir = data_dir + 'brainusr/datasets/' + data_sub_dir
index_fn = jag_dir + 'index.txt'
mapping_fn = jag_dir + 'id_mapping.txt'
output_base = 'sample_list'

import os

# Get random seed for the next run of select_samples
random_seed = 1 #will be incremented 
def get_random_seed() :
  global random_seed
  if repeat_time :
    random_seed += 1
  else :
    random_seed += int(str((a - int(a))).split('.')[1])
  return str(random_seed)

# Make directory, if it doesn't exist
def make_dir(d) :
  print("PY: Making output directory (if it doesn\'t exist): " + d) 
  r = os.system('mkdir -p ' + d)
  if r :
    print('PY: failed to make output directory: ' + d + '- can\'t continue')
    exit(9)

num_lists = 1 
def run_partitioner(index_fn, mapping_fn, num_samples_per_list, num_lists, output_dir, output_base_fn, random_seed) :
  seed = str(random_seed)
  cmd = exe + ' --index_fn=' + index_fn \
            + ' --mapping_fn=' + mapping_fn \
            + ' --num_samples_per_list=' + str(num_samples_per_list) \
            + ' --num_lists=' + str(num_lists) \
            + ' --output_dir=' + output_dir \
            + ' --output_base_fn=' + output_base_fn \
            + ' --random_seed=' + seed

  print('PY: about to run: ' + cmd)
  r = os.system(cmd)
  if r != 0 :
    print('PY: cmd failed: ' + cmd)
    print('PY: Can\'t continue')
    exit(-1)

make_dir(output_dir)

run_partitioner(index_fn, mapping_fn, validation_sample_count, num_lists, output_dir, output_base, get_random_seed())

# This is brittle!
index_fn = output_dir + '/t_exclusion_' + output_base + '_bar' #XX

a = open(index_fn)
header = a.readline()
header = a.readline()
t = header.split()
num_samples = int(t[0])


#subtract 10 so random number generator doesn't spin ...
multi_sample_size = int((num_samples / num_multi_lists) - 10)

print('PY: num samples in ' + index_fn + ' is ' + str(num_samples))
print('PY: num samples in multi lists is ' + str(multi_sample_size))
print('PY: num_list: ' + str(num_lists))

for j in range(num_trials) :
  next_fn = output_dir + '/' + str(j)
  make_dir(next_fn)
  run_partitioner(index_fn, mapping_fn, multi_sample_size, num_multi_lists, next_fn, output_base, get_random_seed())

cmd = 'find ' + output_dir + ' -type f | grep -v bar | grep -v exclusion | grep -v index | grep ' + output_base + ' > /tmp/erase_me'
r = os.system(cmd)
if r :
  print('PY: cmd failed: ' + cmd)
  exit(9)

# Permute the sample_list entries
import random
v = open('/tmp/erase_me').readlines()
for line in v :
  b = open(line[:-1]).readlines()
  print('PY: opening for write: ' + line[:-1])
  out = open(line[:-1] + '_permuted', 'w')
  for j in range(3) :
    out.write(b[j])
  p = [random.randint(3,len(b)-1) for _ in range(len(b)-3)]
  for j in range(3, len(b)) :
    out.write(b[p[j-3]])
  print('PY: len(b): '+str(len(b))+' len(p): ' + str(len(b)))
  out.close();

cmd = 'find ' + output_dir + ' | grep -v bar | grep -v exclusion > /tmp/erase_me'
cmd = 'find ' + output_dir + ' | grep permuted  > /tmp/erase_me'
r = os.system(cmd)
if r :
  print('PY: cmd failed: ' + cmd)
  exit(9)

v = open('/tmp/erase_me').readlines()
out = open(output_dir + '/sample_list_index.txt', 'w')
out.write('The following is a list inclusion filenames for use in experiments:\n')
for line in v : 
  out.write(line)
out.close()

print('\n\nPY: SEE: ' + output_dir + '/sample_list_index.txt for list of sample filenames\n')
