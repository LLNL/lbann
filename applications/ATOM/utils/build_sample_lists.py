#!/usr/tce/bin/python

import sys
import random

def usage() :
  print 'usage:', sys.argv[0], 'input_file output_file_base_name [random_seed]'
  print '''
  where: input_filename contains:
    1st line: num_trainers
    2nd line: data base directory
    remaining lines: smiles_cvs_filename n_samples_total n_samples_per_trainer reuse

    where: 
      'n_samples_total' is the number of samples (SMILES strings) in the csv file
      'n_samples_per_trainer' is the number of samples from the cvs file to be used for each trainer
      'reuse' is '1' or '0'; if '1' then the same set of samples (sample_ids) will be used for each trainer; else, the sets will be unique
  '''

if len(sys.argv) < 3 :
  usage()
  exit(9)

if len(sys.argv) == 4 :
  random.seed( int(sys.argv[3]) )

output_base = sys.argv[2][:-1]

a = open(sys.argv[1]).readlines()
idx = 0
while a[idx][0] == '#' :
  idx += 1

t = a[idx].split()
idx += 1
num_trainers = int(t[0])

t = a[idx].split()
idx += 1
base_data_dir = t[0]

inputs = {}
for j in range(idx, len(a)) :
  t = a[j].split()
  #total samples, num_to_use, reuse
  inputs[t[0]] = (int(t[1]), int(t[2]), int(t[3])) 


# deal with subsets common to all sample lists
common = {}
for fn in inputs.keys() :
  if inputs[fn][2] == 1 :
    print 'using a common set of', inputs[fn][1],  'random indices from', fn, 'for all sample lists'
    indices = set()
    while len(indices) < inputs[fn][1] :
      indices.add(random.randint(0, inputs[fn][0]-1))
    common[fn] = indices
    inputs[fn] = None

#total num samples in each sample list
num_samples = 0
for fn in common :
  num_samples += len(common[fn])
for fn in inputs :
  if inputs[fn] != None :
    num_samples += inputs[fn][1]
    print 'using a unique set of', inputs[fn][1], 'random indices from', fn, 'for each sample_list'

print '\nnum samples per sample_list:', num_samples

# Generate the lists
used = {}
for fn in inputs :
  used[fn] = set()

for k in range(num_trainers) :
  fn = output_base + '_' + str(k) + '.txt'
  print 'writing:', fn
  out = open(fn, 'w')
  out.write('CONDUIT_HDF5_INCLUSION\n')
  out.write(str(num_samples) + ' 0 ' + str(len(inputs)) + '\n')
  out.write('/p/gpfs1/brainusr/datasets/atom/combo_enamine1613M_mpro_inhib\n')
  for fn in inputs.keys() :
    if inputs[fn] == None :
      out.write(fn + ' ' + str(len(common[fn])) + ' 0')
      for x in common[fn] :
        out.write(' ' + str(x))
      out.write('\n')
    else :
      num_to_use = inputs[fn][1]
      total = inputs[fn][0]
      out.write(fn + ' ' + str(num_to_use) + ' 0')
      useme = set()
      print 'selecting', num_to_use, 'random indices from', total, 'indices'
      while len(useme) < num_to_use :
        r = random.randint(0, total-1)
        if r not in used[fn] :
          useme.add(r)
          used[fn].add(r)
      for id in useme :
        out.write(' ' + str(id))
      out.write('\n')
  out.close()
