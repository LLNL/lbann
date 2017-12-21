import csv
import os
import re
import sys
import time
import unittest

def model(model_name, dir_name):
  return '--model=%s/model_zoo/models/%s/model_%s.prototext' % (dir_name, model_name, model_name)

def reader(reader_name, dir_name):
  return '--reader=%s/model_zoo/data_readers/data_reader_%s.prototext' % (dir_name, reader_name)

def optimizer(optimizer_name, dir_name):
  return '--optimizer=%s/model_zoo/optimizers/opt_%s.prototext' % (dir_name, optimizer_name)

def run_lbann(dir_name, executable, model_name, reader_name, optimizer_name, output_file_name):
  m = model(model_name, dir_name)
  r = reader(reader_name, dir_name)
  o = optimizer(optimizer_name, dir_name)
  # N => number of nodes.      
  # p => partition.
  # t => timeout period, in minutes.
  # n => number of processes to run. MPI Rank.
  # n / procs_per_model = how many models should be made. (It must be true that n >= procs_per_model).
  # num_epochs => number of epochs.
  # data_reader_percent => how much of the data to use.
  if model_name == 'alexnet':
    command = 'salloc -N 16 -p pbatch -t 600 srun -n 32 %s %s %s %s --num_epochs=20 --data_reader_percent=0.10 > %s' % (executable, m, r, o, output_file_name)
  elif model_name == 'lenet_mnist':
    command = 'salloc -N 1 -p pbatch -t 60 srun -n 2 %s %s %s %s --num_epochs=5 > %s' % (executable, m, r, o, output_file_name)
  else:
    raise Exception('Invalid model: %s' % model_name)

  print('About to run: %s' % command)
  print('%s began waiting in the queue at ' % model_name + time.strftime('%H:%M:%S', time.localtime()))
  value = os.system(command)
  print('%s finished at ' % model_name + time.strftime('%H:%M:%S', time.localtime()))
  if value != 0:
    output_file = open(output_file_name, 'r')
    for line in output_file:
      print('%s: %s' % (output_file_name, line))
    raise Exception('Model %s crashed' % model_name)

def get_performance(output_file_name, should_log=False):
  output_file = open(output_file_name, 'r')
  performance_dict = {}

  for line in output_file:

    if should_log:
      print('%s: %s' % (output_file_name, line))

    # Check if line is reporting model results
    m = re.search('^(Model [0-9][0-9]*)', line)
    if m:
      model = m.group()

      # Add model to dictionary if needed
      if model not in performance_dict.keys():
        performance_dict[model] = {
          'accuracies' : [],
          'epoch_times' : [],
          'mean_minibatch_times' : [],
          'maxs' : [],
          'mins' : [],
          'stdevs' : []
          }

      # Check if line reports epoch run time
      m = re.search('training epoch [0-9]* run time : ([^s]*)s', line)
      if m:
        performance_dict[model]['epoch_times'].append(float(m.group(1)))

      # Check if line reports mini-batch time statistics
      m = re.search('training epoch [0-9]* mini-batch time statistics : ([^s]*)s? mean, ([^s]*)s? max, ([^s]*)s? min, ([^s]*)s? stdev', line)
      if m:
        performance_dict[model]['mean_minibatch_times'].append(float(m.group(1)))
        performance_dict[model]['maxs'].append(float(m.group(2)))
        performance_dict[model]['mins'].append(float(m.group(3)))
        performance_dict[model]['stdevs'].append(float(m.group(4)))

      # Check if line reports test accuracy
      m = re.search('test categorical accuracy : ([^%]*)%', line)
      if m:
        performance_dict[model]['accuracies'].append(float(m.group(1)))

  output_file.close()
  if should_log:
    print(performance_dict)
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

def run_tests(performance, model_name, dir_name, should_log=False):
  expected_times = csv_to_dict('%s/bamboo/integration_tests/performance_tests/expected_performance.csv' % dir_name)
  errors = []
  numbers = []
  for model_num in performance.keys():
    p = performance[model_num]
    e = expected_times[model_name]
    for epoch_time in p['epoch_times']:
      if epoch_time > e['max_epoch_time']:
        errors.append('%.2f > %.2f %s %s max_epoch_time' % (epoch_time, e['max_epoch_time'], model_name, model_num))
      numbers.append('%.2f %s %s epoch_time' % (epoch_time, model_name, model_num))
    for mean_minibatch_time in p['mean_minibatch_times']:
      if mean_minibatch_time > e['max_mean_minibatch_time']:
        errors.append('%.2f > %.2f %s %s max_mean_minibatch_time' % (mean_minibatch_time, e['max_mean_minibatch_time'], model_name, model_num))
      numbers.append('%.2f %s %s mean_minibatch_time' % (mean_minibatch_time, model_name, model_num))
    for max_time in p['maxs']:
      if max_time > e['max_max_time']:
        errors.append('%.2f > %.2f %s %s max_max_time' % (max_time, e['max_max_time'], model_name, model_num))
      numbers.append('%.2f %s %s max_time' % (max_time, model_name, model_num))
    for min_time in p['mins']:
      if min_time > e['max_min_time']:
        errors.append('%.2f > %.2f %s %s max_min_time' % (min_time, e['max_min_time'], model_name, model_num))
      numbers.append('%.2f %s %s min_time' % (min_time, model_name, model_num))
    for stdev in p['stdevs']:
      if stdev > e['max_stdev']:
        errors.append('%.2f > %.2f %s %s max_stdev' % (stdev, e['max_stdev'], model_name, model_num))
      numbers.append('%.2f %s %s stdev' % (stdev, model_name, model_num))
    for accuracy in p['accuracies']:
      if accuracy < e['min_accuracy']:
        errors.append('%.2f < %.2f %s %s min_accuracy' % (accuracy, e['min_accuracy'], model_name, model_num))
      numbers.append('%.2f %s %s accuracy' % (accuracy, model_name, model_num))
  print('Errors for: %s (%d)' % (model_name, len(errors)))
  for error in errors:
    print(error)
  if should_log:
    print('All numbers for: %s (%d)' % (model_name, len(numbers)))
    for number in numbers:
      print(number)
  assert errors == []

def skeleton(dir_name, executable, model_name, reader_name, should_log=False):
  output_file_name = '%s_output.txt' % model_name
  run_lbann(dir_name, executable, model_name, reader_name, 'adagrad', output_file_name)
  performance = get_performance(output_file_name, should_log)
  run_tests(performance, model_name, dir_name, should_log)
