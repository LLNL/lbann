import sys
import os
import random

def shuffle(fn) :
  fn2 = fn + '.shuffled'
  a = open(fn).readlines()
  b = open(fn2, 'w')
  b.write(a[0])
  b.write(a[1])
  b.write(a[2])
  c = a[3:]
  n = len(c)
  r = set()
  r_idx = []
  for y in range(n) :
    while True :
      y = random.randint(0, n-1)
      if y not in r :
        r.add(y)
        r_idx.append(y)
      if len(r) == n :
        break
  for j in range(len(c)) :
    b.write(c[r_idx[j]])
  b.close()
  print 'wrote:', fn2

#====================================================================
if len(sys.argv) != 4 :
  print 'usage:', sys.argv[0], 'base_dir num_sample_lists sample_list_base_name'
  print 'example: python', sys.argv[0], '/p/lustre2/brainusr/datasets/10MJAG/1M_A/select_samples_test/another_dir 10 my_samples.txt',
  exit(9)

dir = sys.argv[1]
n = int(sys.argv[2])
base_fn = sys.argv[3]

for j in range(n) :
  fn = dir + '/t' + str(j) + '_' + base_fn
  shuffle(fn)

