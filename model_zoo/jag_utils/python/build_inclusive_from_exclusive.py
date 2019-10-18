import sys

if len(sys.argv) != 4 :
  print 'usage:', sys.argv[0], 'index_fn id_mapping_fn output_fn'
  exit(9)

a = open(sys.argv[1])
a.readline()
header = a.readline()
dir = a.readline()

#build map: filename -> set of bad samples
mp = {}
mp_good = {}
mp_bad = {}
for line in a :
  t = line.split()
  mp[t[0]] = set()
  mp_good[t[0]] = t[1]
  mp_bad[t[0]] = t[2]
  for id in t[3:] :
    mp[t[0]].add(id)

a.close()

out = open(sys.argv[3], 'w')
out.write('CONDUIT_HDF5_INCLUSION\n')
out.write(header)
out.write(dir)

a = open(sys.argv[2])
bad = 0
for line in a :
  t = line.split()
  fn = t[0]
  out.write(fn + ' ' + mp_good[fn] + ' ' + mp_bad[fn] + ' ')
  for id in t[1:] :
    if id not in mp[fn] :
      out.write(id + ' ')
    else :
      bad += 1
  out.write('\n')    

out.close()
print header
print 'num found bad:', bad
