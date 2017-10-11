#!/usr/bin/python
import os
import sys
import common

this_dir = os.path.dirname(os.path.realpath(__file__))

os.system('find ' + this_dir + '/.. -name "runme*.py" > ok_to_erase_me')
a = open('ok_to_erase_me').readlines()
os.system('rm -f ok_to_erase_me')


'''
for line in a :
  x = line[:-1]
  x += ' --exit_after_setup'
  os.system(x)
'''

for line in a :
  j = line.rfind('/')
  assert(j != -1)
  work_dir = line[:j]
  c = 'cd ' + work_dir + '; runme.py --exit_after_setup > ' + this_dir + '/ok_to_erase_me'
  os.system(c)
  cmd = open('ok_to_erase_me').readline()

  t = cmd.split()
  cmd = ''
  for x in t :
    if x.find('--model=') != -1 :
      t2 = x.split('=')
      cmd += '--model=' + work_dir + '/' + t2[1] + ' '
    elif x.find('--reader=') != -1 :
      t2 = x.split('=')
      cmd += '--reader=' + work_dir + '/' + t2[1] + ' '
    elif x.find('--optimzer=') != -1 :
      t2 = x.split('=')
      cmd += '--optimizer=' + work_dir + '/' + t2[1] + ' '
    else :
      cmd += x + ' '

  os.system('rm -f ok_to_erase_me')

  #sys.stderr.write('\n============================================================\n') 
  sys.stderr.write('echo "about to run: ' + cmd + '"\n\n')
  sys.stderr.write(cmd + '\n\n')
  #sys.stderr.write('about to run: ' + e + '\n\n')
  #os.system(e)
