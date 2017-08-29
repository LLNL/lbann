import socket
import sys
import datetime
import os

# This list contains cmd line options that are used by python scrips herein;
# these options are *not* passed to the lbann cmd line
script_options = ['--nodes', '--bindir', '--tasks', '--ssd', '--data_reader', '--optimizer', '--no_submit', '--time_limit']

def runme(cmd, msg=None) :
  '''Submits a job to the shell. If the job fails, prints diagnostic message
     to stderr, then exits.
  '''
  if msg :
    sys.stderr.write(msg + '\n')
  result = os.system(cmd)
  if result :
    sys.stderr.write('This cmd failed to execute: ' + cmd + '\n')
    exit(9)

def stage_data_to_ssd(filename=None) :
  '''If the cmd line contains --ssd, calling this function will stage
     data files (copy then expand tarball(s)) to /l/ssd
  '''
  for n in sys.argv :
    if n.find('-ssd') != -1 :
      print
      print
      print 'stage_data_to_ssd() needs to be implemented'

def lbann_options() :
  '''Parses the cmd line; returns a list of options that are to be
     passed to the lbann cmd line. Specifically, excludes arguments
     that appear in script_options [].
  '''
  o = ''
  for j in range(1, len(sys.argv)) :
    n = sys.argv[j]
    keep = True
    for p in script_options :
      if n.find(p) != -1 :
        keep = False
    if keep :
      o += n + ' '
  return o

def hostname() :
  ''' Returns the hostname, without number, e.g, if hostname is
      is "surface86" returns "surface"
  '''
  a = socket.gethostname()
  while a[-1].isdigit() :
    a = a[:-1]
  return a


def bindir() :
  '''Returns the directory that contains the lbann executable. This may be
     specified on the command line via the --bindir=<string> option; else,
     returns the relative directory: '../../..build/<hostname()>.llnl.gov/model_zoo
  '''
  b = '../../../build/' + hostname() + '.llnl.gov/model_zoo'
  for n in sys.argv :
    if n.find('--bindir=') != -1 :
      t = n.split('=')
      b = t[0][2:]
  return b

def help(model = False, data_reader = False, optimizer = False) :
  '''Returns a multi-line help message string '''
  h = ''
  if model != False :
    h += 'model filename:       ' + model + '\n'
  if data_reader != False :
    h += 'data_reader filename: ' + data_reader + '\n'
  if optimizer != False :
    h += 'optimizer filename:   ' + optimizer + '\n'

  h += '''
  run.py constructs a script titled "slurm_script.sh" then invokes msub.

  The following command line options are used by run.py;
  to view command line options that are used by LBANN,
  run: $ run.py --help

  REQUIRED OPTIONS:
  =================
  --nodes=<int>     The number of nodes
  --tasks=<int>     The number of tasks per node; 
                    total number of cores is --nodes * --tasks_per_node

  OPTIONAL OPTIONS:
  =================
  --this_dir   slurm_script.sh and output from the slurm job appear in this 
               directory; default it to place output in a directory with the
               timestamp: $(date +%Y%m%d_%H%M%S)

  --no_submit  Construct slurm_script.sh but do not invoke $ msub slurm_script.sh

  --time_limit=<string>  Default is 12:00:00

  --bindir=<string>   Specifies the build directory; default is:
                         <some_path_to>/lbann/build
                      You only need to specify  if your build is elsewhere

  --ssd   Stage data to /l/ssd; default: read data from
          lscratch file systems. If you use this option
          ensure you also use --data_reader=<string>

  --data_reader=<string>  Specify an alternative/appropriate reader;
                          see data_reader subdirectory for available names. 
                          To use "data_reader_mnist.prototext" you would
                          specify: --data_reader=mnist

  --optimizer=<string>    specify an alternative optimizer
                          see optimizer subdirectory for available names.
                          To use "opt_rmsprop.prototext" you would
                          specify: --optimizer=rmsprop
  '''
  return h

def run_cmd(model, data_reader, optimizer) :
  '''Returns a command line that will run the lbann executable using prototext
     files. The returned string is something like: 
       srun --node=<int> --ntasks_per_node=<int> <path to>/lbann --todoTODO
  '''
  for n in sys.argv :
    if n == '--help' or n == '-h' :
      cmd = bindir() + '/lbann ' + lbann_options() \
            + ' --model=' + model                  \
            + ' --reader=' + data_reader           \
            + ' --optimizer=' + optimizer          \
            + ' --help'
      os.system(cmd)
      return

  nnodes = 0
  ntasks_per_node = 0
  for n in sys.argv :
    if n.find('--nodes=') != -1 :
      t = n.split('=')
      nnodes = t[1]
    if n.find('--tasks=') != -1 :
      t = n.split('=')
      ntasks_per_node = t[1]
  if nnodes == 0 or ntasks_per_node == 0 :
    print '\n\nERROR: you must specify both --nodes=<int> and --tasks=<int>'
    print '-------------------------------------------------------------------------------'
    print help()
    print '-------------------------------------------------------------------------------'
    print '\nERROR: you must specify both --nodes=<int> and --tasks=<int>\n'
    exit(9)
  return 'srun --nodes=' + str(nnodes)                  \
         + ' --ntasks-per-node=' + str(ntasks_per_node) \
         + ' ' + bindir() + '/lbann ' + lbann_options() \
         + ' --model=' + model                          \
         + ' --reader=' + data_reader                   \
         + ' --optimizer=' + optimizer

def build_and_submit_slurm_script(model, data_reader, optimizer) :
  '''Constructs and writes to file: slurm_script.sh. The script is also
     submitted ("msub slurm_script.sh") unless the -no_submit option
     appears on the command line. Various informational messages may be
     written to stdout; these include all 'srun ...' commands'
  '''
  print run_cmd(model, data_reader, optimizer) 
  #where = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())  
  '''
  for n in sys.argv :
    if n == 'this_dir' :
      where = '.'

  out = open(where + '/slurm_script.sh', 'w')
  out.write('#!/bin/bash\n\n')
  out.write('# ======== Experiment parameters ========\n')
  out.write('# Directory: ' + where + '\n')

  out.close()
  '''

'''
print hostname()
print bindir()
print 'opts:', lbann_options()
print base_cmd()
print
print help(model = 'xyz', data_reader = 'imagenet')
stage_data_to_ssd()
'''
