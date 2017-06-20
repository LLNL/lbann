#!/usr/bin/python
import os
from sys import *

usage = '''
usage: %s dir=<string> ext=.h,.hpp,.cpp exec=/usr/gapps/brain/bin/astyle_surface

function: recursively search the directory structure rooted at 'dir'
          and runs atyle. 

arguments:
  dir=<string>  - required
  ext=h,hpp,cpp - optional; h,hpp,cpp are the defaults, and indictes
                  that formatting will only occurr for files ending in
                  .h .hpp .cpp
  exec=<string> - optional; pathname of the executable. For catalyst use:
                   exec=/usr/gapps/brain/bin/astyle_catalyst
''' % argv[0]

opts = ' -s2 --style=google --add-braces --convert-tabs --align-pointer=name  --align-reference=type'

def error(msg) :
  print usage
  print '========================================================'
  print 'ERROR:'
  print msg
  exit(9)

dir = 'none'
ext = ['.h', '.hpp', '.cpp']
cmd = '/usr/gapps/brain/bin/astyle_surface'

for i in range(1, len(argv)) :
  j = argv[i]
  if j.find('=') != -1 :
    t = j.split('=')
    if t[0] == 'dir' :
      dir = t[1]
    elif t[0] == 'cmd' :
      cmd = t[1]
    elif t[0] == 'ext' :
      ext = t[1].split(',')
    else :
      error('one or more of your params was not recognized, or is ill-formed')

  else :
    error("all params should be of the form <string>=<string>; you passed a param that did not contain '='")
    
if not os.path.isfile(cmd) :
  error('this file does not exist: ' +  cmd)

if not os.path.isdir(dir) :
  error('this directory does not exist: ' + dir)

print ext
print
print
for e in ext :
  runme = 'find ' + dir + ' -name "*' + e + '" -print -exec ' + cmd + ' ' + opts + ' {} \;'
  os.system(runme)
