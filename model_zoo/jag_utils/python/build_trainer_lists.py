#!/usr/bin/env python
import os
import subprocess
import sys
import random
import time


def runme(cmd) :
  print 'about to run system call:', cmd
  t = cmd.split()
  r = subprocess.check_call(t)


if len(sys.argv) < 8:
  print '\nusage:', sys.argv[0], 'index_fn sample_mapping_fn num_samples num_lists output_dir output_base_name random_seed [HOST]'
  print 'function: creates "num_lists" sample lists from index_fn;'
  print '          each list will contain "num_samples." Each list is printed'
  print '          to a separate file'
  print
  print 'if your environment doesn\'t contain HOST (e.g: $echo $HOST pascal83) then you'
  print 'can specify HOST as the final cmd line param'
  print
  print 'example invocation, lassen:'
  print '   $ build_trainer_lists.py /p/gpfs1/brainusr/datasets/10MJAG/1M_B/index.txt /p/gpfs1/brainusr/datasets/10MJAG/1M_B/id_mapping.txt 10000 4 /p/gpfs1/brainusr/datasets/10MJAG/1M_B sample_list 42\n'
  print
  print 'example invocation, lustre:'
  print '   $ build_trainer_lists.py /p/lscratchh/brainusr/datasets/10MJAG/1M_B/index.txt /p/lscratchh/brainusr/datasets/10MJAG/1M_B/id_mapping.txt 10000 4 /p/lscratchh/brainusr/datasets/10MJAG/1M_B sample_list 42\n'
  exit(9)

# defaults; because who doesn't use gnu?
build = 'Release'
compiler = 'gnu'

# this will fail if we're not running in an lbann repo
lbann_dir = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])[:-1]

# get cluster name
host = ''
if len(sys.argv) == 9 :
  host = sys.argv[8]
else :
  try :
    host = os.environ['HOST']
  except :
    print '\nYour environment does not appear to contain the HOST variable;'
    print 'therefore, please specify HOST as the final argument on the cmd line'
    exit(9)

cluster = ''
for x in os.environ['HOST'] :
  if not x.isdigit() :
    cluster += x

index_fn = sys.argv[1]
mapping_fn = sys.argv[2]
num_samples = sys.argv[3]
num_lists = int(sys.argv[4])
output_dir = sys.argv[5]
output_base_name = sys.argv[6]
seed = sys.argv[7]

# get path to the c++ executable
exe = lbann_dir + '/build/' + compiler + '.' + build + '.' + cluster \
    + '.llnl.gov/lbann/build/model_zoo/jag_utils/select_samples'
cur_dir = os.getcwd()

# seed the random number generator
random.seed(seed)

first_fn = output_dir + '/t0_' + output_base_name + '.txt'
bar_fn = output_dir + '/t_' + output_base_name + '.txt_bar'

print 'constructing trainer file # 0 ... please wait ...'
cmd = exe + ' --index_fn=' + index_fn + ' --sample_mapping_fn=' + mapping_fn \
          + ' --num_samples=' + num_samples + ' --output_fn=' + first_fn    \
          + ' --random_seed=' + seed
runme(cmd)

cmd = 'mv ' + first_fn + '_bar ' + bar_fn
runme(cmd)

filenames = []
filenames.append(first_fn)

for j in range(1, num_lists) :
  fn = output_dir + '/t' + str(j) + '_' + output_base_name + '.txt'
  print 'constructing trainer file #', j, '... please wait ...'

  cmd = exe + ' --index_fn=' + bar_fn + ' --sample_mapping_fn=' + mapping_fn \
            + ' --num_samples=' + num_samples + ' --output_fn=' + fn    \
            + ' --random_seed=' + seed
  runme(cmd)
  filenames.append(fn)

  cmd = 'mv ' + fn + '_bar ' + bar_fn
  runme(cmd)
filenames.append(bar_fn)

os.system('chgrp brain ' + output_dir + '/*')
os.system('chmod 660 ' + output_dir + '/*')

print
print '=================================================================\n'
print 'generated these files:'
for f in filenames :
  print f

