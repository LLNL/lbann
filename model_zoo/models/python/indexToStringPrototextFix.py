#!/usr/bin/python
from sys import *

usage = '''
usage: %s input_fn
where: input_fn is a model_*.prototext file name
function: changes 'index' to 'name' and puts quotes around the
          index and name fields (thereby converting them from ints to strings)
output: input_fn is saved to input_fn.bak and a new input_fn
        is written with the changes.
''' % argv[0]

if len(argv) != 2 :
  print usage
  exit(9)

'''
if argv[1][:6] != 'model_' :
  print usage
  print '======================================================================='
  print 'ERROR: input_fn must be of the form: model_*.prototext'
  exit(9)
'''

#write backup file of original
a = open(argv[1]).readlines()
backup_fn = argv[1] + '.bak'
out = open(backup_fn, 'w')
for line in a :
  out.write(line)
out.close()
stderr.write('\nwrote backup file: ' + backup_fn + '\n')

out = open(argv[1], 'w')
for line in a :
  if line.find('index:') != -1 :
    whitespace = line.find('index')
    n = line.split()
    out.write(' ' * whitespace + 'name: "' + n[1] + '"\n')
  elif line.find('parent:') != -1 and line.find('"') == -1 :
    whitespace = line.find('parent')
    n = line.split()
    out.write(' ' * whitespace + 'parent: "' + n[1] + '"\n')
  else :
    out.write(line)
out.close()
