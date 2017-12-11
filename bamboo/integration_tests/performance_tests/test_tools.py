import csv
import os
import re
import sys
import time
import unittest

# This cannot be 'test_model'. If it were, then pytest would try to run it.
def model_test(model_name, dir_name):
  return '--model=%s/model_zoo/tests/model_%s.prototext' % (dir_name, model_name)

def model(model_name, dir_name):
  return '--model=%s/model_zoo/models/%s/model_%s.prototext' % (dir_name, model_name, model_name)

def reader(reader_name, dir_name):
  return '--reader=%s/model_zoo/data_readers/data_reader_%s.prototext' % (dir_name, reader_name)

def optimizer(optimizer_name, dir_name):
  return '--optimizer=%s/model_zoo/optimizers/opt_%s.prototext' % (dir_name, optimizer_name)

def run_lbann(model_name, reader_name, optimizer_name, output_file_name, executable, dir_name, num_nodes=1, timeout=60, num_processes=2, num_epochs=5, procs_per_model=1):
  if model_name == 'mnist_distributed_io':
    m = model_test(model_name, dir_name)
  else:
    m = model(model_name, dir_name)
  r = reader(reader_name, dir_name)
  o = optimizer(optimizer_name, dir_name)
  # N => number of nodes                                                         
  # p => partition
  # t => timeout period, in minutes
  # n => number of processes to run. MPI Rank.
  # n / procs_per_model = how many models should be made. (n >= procs_per_model)
  # num-epochs => The number of epochs
  command = 'salloc -N %d -p pbatch -t %d srun -n %d %s %s %s %s --num_epochs=%d --procs_per_model=%d > %s' % (num_nodes, timeout, num_processes, executable, m, r, o, num_epochs, procs_per_model, output_file_name)
  print('Began %s at ' % model_name + time.strftime('%H:%M:%S', time.localtime()))
  value = os.system(command)
  print('Ended %s at ' % model_name + time.strftime('%H:%M:%S', time.localtime()))
  if value != 0:
    raise Exception('Model %s crashed' % model_name)

def get_performance(output_file_name):
  output_file = open(output_file_name, 'r')
  performance_dict = {}

  for line in output_file:

    # Check if line is reporting model results
    m = re.match('^(Model [0-9][0-9]*)', line)
    if m:
      model = m.group()

      # Add model to dictionary if needed
      if model not in performance_dict.keys():
        performance_dict[model] = {
          'accuracies' : [],
          'epoch_times' : [],
          'mean_minibatch_times' : [],
          'mins' : [],
          'maxs' : [],
          'stdevs' : []
          }

      # Check if line reports epoch run time
      m = re.match('training epoch [0-9]* run time : ([^s]*)s', line)
      if m:
        performance_dict[model]['epoch_times'].append(float(m.group(1)))

      # Check if line reports mini-batch time statistics
      m = re.match('training epoch [0-9]* mini-batch time statistics : '
                   '([^s]*)s mean, ([^s]*)s min, ([^s]*)s max, ([^s]*)s stdev', line)
      if m:
        performance_dict[model]['mean_minibatch_times'].append(float(m.group(1)))
        performance_dict[model]['mins'].append(float(m.group(2)))
        performance_dict[model]['maxs'].append(float(m.group(3)))
        performance_dict[model]['stdevs'].append(float(m.group(4)))

      # Check if line reports test accuracy
      m = re.match('test categorical accuracy : ([^%]*)%', line)
      if m:
        performance_dict[model]['accuracies'].append(float(a.group(1)))

  output_file.close()
  return performance_dict


def csv_to_dict(csv_path):
  with open(csv_path, 'r') as csv_file:
    reader = csv.reader(csv_file, skipinitialspace=True)
    keys = reader.next()
    expected_times = {}
    for row in reader:
      model = row[0]
      expected_times[model] = dict(zip(keys[1:], map(float, row[1:])))
  return expected_times

def run_tests(performance, model_name, dir_name):
  expected_times = csv_to_dict('%s/bamboo/integration_tests/performance_tests/expected_performance.csv' % dir_name)
  errors = []
  for model_num in performance.keys():
    p = performance[model_num]
    e = expected_times[model_name]
    for epoch_time in p['epoch_times']:
      if epoch_time > e['max_epoch_time']:
        errors.append('%.2f > %.2f %s %s max_epoch_time' % (epoch_time, e['max_epoch_time'], model_name, model_num))
    for mean_minibatch_time in p['mean_minibatch_times']:
      if mean_minibatch_time > e['max_mean_minibatch_time']:
        errors.append('%.2f > %.2f %s %s max_mean_minibatch_time' % (mean_minibatch_time, e['max_mean_minibatch_time'], model_name, model_num))
    for min_time in p['mins']:
      if min_time > e['max_min_time']:
        errors.append('%.2f > %.2f %s %s max_min_time' % (min_time, e['max_min_time'], model_name, model_num))
    for max_time in p['maxs']:
      if max_time > e['max_max_time']:
        errors.append('%.2f > %.2f %s %s max_max_time' % (max_time, e['max_max_time'], model_name, model_num))
    for stdev in p['stdevs']:
      if stdev > e['max_stdev']:
        errors.append('%.2f > %.2f %s %s max_stdev' % (stdev, e['max_stdev'], model_name, model_num))
    for accuracy in p['accuracies']:
      if accuracy < e['min_accuracy']:
        errors.append('%.2f < %.2f %s %s min_accuracy' % (accuracy, e['min_accuracy'], model_name, model_num))
  print "Errors for: %s" % model_name
  for error in errors:
    print error
  assert errors == []

def mnist_distributed_io_skeleton(executable, dir_name):
  model_name = 'mnist_distributed_io'
  output_file_name = '%s_output.txt' % model_name
  run_lbann(model_name=model_name, reader_name='mnist', optimizer_name='adagrad', output_file_name=output_file_name, executable=executable, dir_name=dir_name)
  performance = get_performance(output_file_name)
  run_tests(performance, model_name, dir_name)

def alexnet_skeleton(executable, dir_name):
  model_name = 'alexnet'
  output_file_name = '%s_output.txt' % model_name
  run_lbann(model_name=model_name, reader_name='imagenet', optimizer_name='adagrad', output_file_name=output_file_name, executable=executable, dir_name=dir_name)
  performance = get_performance(output_file_name)
  run_tests(performance, model_name, dir_name)

def resnet50_skeleton(executable, dir_name):
  model_name = 'resnet50'
  output_file_name = '%s_output.txt' % model_name
  run_lbann(model_name=model_name, reader_name='imagenet', optimizer_name='adagrad', output_file_name=output_file_name, executable=executable, dir_name=dir_name, num_epochs=1, procs_per_model=2)
  performance = get_performance(output_file_name)
  run_tests(performance, model_name, dir_name)
