import sys
import random

if len(sys.argv) != 5 :
  print '\nusage:', sys.argv[0], ' master_index_fn num_samples output_fn random_seed\n'
  print 'outputs: output_fn, which is formatted per CONDUIT_HDF5_INCLUSION'
  print '         output_fn_bar, which is formatted per CONDUIT_HDF5_EXCLUSION\n'
  exit(9)

#===========================================================================
def write_output(fn, header) :
  tmp = open(fn).readlines()
  out = open(fn, 'w')
  out.write(header + '\n')

  num_files = len(tmp)
  n_good = 0
  n_bad = 0
  for line in tmp :
    t = line.split()
    n_good += int(t[1])
    n_bad += int(t[2])
  out.write(str(n_good) + ' ' + str(n_bad) + ' ' + str(num_files) + '\n')
  out.write(base_dir)
  for line in tmp :
    out.write(line)
  out.close()

#===========================================================================
# parse cmd line, get header and counts from index file
a = open(sys.argv[1])
header = a.readline()
assert(header[:-1] == 'CONDUIT_HDF5_EXCLUSION')
counts = a.readline()
base_dir = a.readline()
t = counts.split()
num_valid = int(t[0])
num_invalid = int(t[1])
num_files = int(t[2])
n_samples = int(sys.argv[2])
assert(n_samples < num_valid)
r_seed = int(sys.argv[4])

print 'input index file has', num_valid, 'valid entries and', num_invalid, 'invalid'

out = open(sys.argv[3], 'w')
out_bar = open(sys.argv[3] + '_bar', 'w')

# generate random indices; note that these are global indices
print 'generating random indices ...'
random.seed(r_seed)
global_indices = set()
while True :
  c = random.randint(0, num_valid - 1)
  if not c in global_indices :
    global_indices.add(c)
  if len(global_indices) % 1000 == 0 : print len(global_indices)/1000, 'K indices generated'
  if len(global_indices) == n_samples : break
print 'DONE! generated', len(global_indices), 'indices'

# loop over each entry from in input index file;
# determine which, if any, local indices will bee
# added to the INCLUSION index
first = 0
i = 0
for line in a :
  i += 1
  if i % 1000 == 0 : print i/1000, 'K input lines processed'
  local_valid_index = 0
  keepme = []
  unused = []
  t = line.split()
  fn = t[0]
  good = int(t[1])
  bad = int(t[2])
  exclude = set()
  for j in range(3, len(t)) :
    exclude.add(int(t[j]))
  for idx in range(good+bad) :
    if idx not in exclude :
      global_idx = local_valid_index + first
      if global_idx in global_indices :
        keepme.append(idx)
      else :
        unused.append(idx)
      local_valid_index += 1
  first += good      

  # only need to add entries for files from which we have selected indices
  if len(keepme) :
    out.write(fn + ' ' + str(len(keepme)) + ' ' + str(len(unused)+bad))
    for idx in keepme : 
      out.write(' ' + str(idx)) 
    out.write('\n')

  # all files go in the exclusion file (todo: if all indices have
  # been used, no need to have an entry for that file)
  out_bar.write(fn + ' ' + str(len(unused)) + ' ' + str(bad+len(keepme)))
  c = list(exclude) + keepme
  c.sort()
  for idx in c : 
    out_bar.write(' ' + str(idx)) 
  out_bar.write('\n')
    
# close tmp files
out.close()
out_bar.close()

# format final inclusion files
write_output(sys.argv[3], 'CONDUIT_HDF5_INCLUSION')
write_output(sys.argv[3] + '_bar', 'CONDUIT_HDF5_EXCLUSION')

# say goodbye
print
print '================================================================'
print 'output written to:', sys.argv[3]
print 'output written to:', sys.argv[3] + '_bar'

