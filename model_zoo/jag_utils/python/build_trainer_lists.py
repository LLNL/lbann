#!/usr/bin/env python
import os
import subprocess
import sys
import random
import time

if len(sys.argv) < 6:
  print 'usage:', sys.argv[0], 'index_fn num_samples num_lists output_dir output_base_fn [random_seed]'
  print 'function: creates "num_lists" sample lists from "index_fn;'
  print '          each list will contain "num_samples.'
  print 'Notes: The "output_dir" will be created if it doesn\'t exit;'
  print '       Output files are of the form: <output_dir>/base_fn_T#_sample_list.txt'
  print '       If "random_see" is not given, the randum number generator will'
  print '       be seeded with int(time.time())'
  exit(9)

# defaults; because who doesn't use gnu?
build = 'Release'
compiler = 'gnu'

# this will fail if we're not running in an lbann repo
lbann_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])[:-1]

# get cluster name
cluster = ''
for x in os.environ['HOST'] :
  if not x.isdigit() :
    cluster += x

# get path to the c++ executable
exe = lbann_dir + '/build/' + compiler + '.' + build + '.' + cluster \
    + '.llnl.gov/lbann/build/model_zoo/jag_utils/select_samples'
cur_dir = os.getcwd()

# seed the random number generator
r_seed = int(time.time())
if (len(sys.argv) == 7) : 
  random.seed(int(sys.argv[6]))

index_fn = sys.argv[1]
n_samples = sys.argv[2]
n_trainers = int(sys.argv[3])
output_dir = sys.argv[4]
lbann_dir = subprocess.check_output(['mkdir', '-p', sys.argv[4]])
output_base_fn = sys.argv[5]

first_fn = output_dir + '/' + output_base_fn + '_T_0_sample_list.txt'
bar_fn =  output_dir + '/' + output_base_fn + '_sample_list_bar.txt'

print 'constructing trainer file # 0 ... this may take up to a minute ...'
r = subprocess.check_output([exe, index_fn, n_samples, first_fn, str(r_seed)])
r = subprocess.check_output(['mv' , first_fn + '_bar', bar_fn])

filenames = []
filenames.append(first_fn)

for j in range(1, n_trainers) :
  fn = output_dir + '/' + output_base_fn + '_T_' + str(j) + '_sample_list.txt'
  print 'constructing trainer file #', j, '... this may take up to a minute ...'
  r = subprocess.check_output([exe, bar_fn, n_samples, fn, str(r_seed)])
  r = subprocess.check_output(['mv' , fn + '_bar', bar_fn])
  filenames.append(fn)
filenames.append(bar_fn)

print
print 'generated these files:'
for f in filenames :
  print f

