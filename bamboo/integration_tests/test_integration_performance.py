import pytest
import os
import common_code

def error_if(f, f_symbol, data_field, actual_values, expected_values, model_name, errors, all_values):
  d = actual_values[data_field]
  if f_symbol == '<':
    # Every time a value is smaller, update archive_value
    archive_value = float('inf')
  elif f_symbol == '>':
    # Every time a value is greater, update archive_value
    archive_value = float('-inf')
  else:
    raise Exception('Invalid Function Symbol %s' % f_symbol)
  for model_id in sorted(d.keys()):
    for epoch_id in sorted(d[model_id].keys()):
      actual_value = d[model_id][epoch_id]
      expected_value = expected_values[model_name][data_field]

      if actual_value == None:
        errors.append('d[%s][%s] == None' % (model_id, epoch_id))
      if expected_value == None:
        errors.append('d[%s]([%s] == None' % (model_id, epoch_id))

      if f(actual_value, expected_value):
        errors.append('%f %s %f %s Model %s Epoch %s %s' % (actual_value, f_symbol, expected_value, model_name, model_id, epoch_id, data_field))
      all_values.append('%f %s Model %s Epoch %s %s' % (actual_value, model_name, model_id, epoch_id, data_field))

      if f(actual_value, archive_value):
        archive_value = actual_value
  return archive_value

def run_tests(actual_performance, model_name, dir_name, should_log, compiler_name, cluster):
  expected_performance = common_code.csv_to_dict('%s/bamboo/integration_tests/expected_values/expected_performance_%s_%s.csv' % (dir_name, compiler_name, cluster))
  errors = []
  all_values = []
  greater_than = lambda x,y: x > y
  less_than = lambda x,y: x < y
  max_run_time = error_if(greater_than, '>', 'training_run_time', actual_performance, expected_performance, model_name, errors, all_values)
  max_mean = error_if(greater_than, '>', 'training_mean', actual_performance, expected_performance, model_name, errors, all_values)
  max_max = error_if(greater_than, '>', 'training_max', actual_performance, expected_performance, model_name, errors, all_values)
  max_min = error_if(greater_than, '>', 'training_min', actual_performance, expected_performance, model_name, errors, all_values)
  max_stdev = error_if(greater_than, '>', 'training_stdev', actual_performance, expected_performance, model_name, errors, all_values)
  min_accuracy = error_if(less_than, '<', 'test_accuracy', actual_performance, expected_performance, model_name, errors, all_values)

  if os.environ['LOGNAME'] == 'lbannusr':
    key = 'bamboo_planKey'
    if key in os.environ:
      plan = os.environ[key]
      if plan in ['LBANN-NIGHTD', 'LBANN-WD', 'LBANN-FOR']:
        archive_file = '/usr/workspace/wsb/lbannusr/archives/%s/%s/%s/performance_%s.txt' % (plan, cluster, compiler_name, model_name)
        with open(archive_file, 'a') as archive:
          archive.write('%s, %f, %f, %f, %f, %f, %f\n' % (os.environ['bamboo_buildNumber'], max_run_time, max_mean, max_max, max_min, max_stdev, min_accuracy))
      else:
        print('The plan %s does not have archiving activated' % plan)
    else:
      print('%s is not in os.environ' % key)
  else:
    print('os.environ["LOGNAME"]=%s' % os.environ['LOGNAME'])

  print('Errors for: %s (%d)' % (model_name, len(errors)))
  for error in errors:
    print(error)
  if should_log:
    print('All values for: %s (%d)' % (model_name, len(all_values)))
    for value in all_values:
      print(value)
  assert errors == []

DATA_FIELDS = [
  'training_run_time',
  'training_mean',
  'training_max',
  'training_min',
  'training_stdev',
  'test_accuracy'
]

def skeleton_performance_lenet_mnist(cluster, dir_name, executables, compiler_name):
  if compiler_name in executables:
    executable = executables[compiler_name]
    model_name = 'lenet_mnist'
    model_folder = 'models/' + model_name
    should_log = False
    actual_performance = common_code.skeleton(cluster, dir_name, executable, model_folder, model_name, DATA_FIELDS, should_log, compiler_name)
    run_tests(actual_performance, model_name, dir_name, should_log, compiler_name, cluster)
  else:
    pytest.skip('default_exes[%s] does not exist' % compiler_name)

def skeleton_performance_alexnet(cluster, dir_name, executables, compiler_name, weekly):
  if weekly:
    if compiler_name in executables:
      executable = executables[compiler_name]
      model_name = 'alexnet'
      model_folder = 'models/' + model_name
      should_log = False
      actual_performance = common_code.skeleton(cluster, dir_name, executable, model_folder, model_name, DATA_FIELDS, should_log)
      run_tests(actual_performance, model_name, dir_name, should_log, compiler_name, cluster)
    else:
      pytest.skip('default_exes[%s] does not exist' % compiler_name)
  else:
    pytest.skip('Not doing weekly testing')

def skeleton_performance_cache_alexnet(cluster, dir_name, executables, weekly, compiler_name):
  if weekly:
    if compiler_name in executables:
      executable = executables[compiler_name]
      model_name = 'cache_alexnet'
      should_log = False
      output_file_name = 'output/%s_output.txt' % model_name
      if (cluster in ['catalyst', 'surface']):
        command = 'salloc %s/bamboo/integration_tests/%s.sh > %s' % (dir_name, model_name, output_file_name)
      else:
        raise Exception("Unsupported Cluster %s" % cluster)
      common_code.run_lbann(command, model_name, output_file_name, should_log)
      actual_performance = common_code.extract_data(output_file_name, DATA_FIELDS, should_log)
      run_tests(actual_performance, model_name, dirname, should_log, compiler_name, cluster)
    else:
      pytest.skip('default_exes[%s] does not exist' % compiler_name)
  else:
    pytest.skip('Not doing weekly testing')

def test_integration_performance_lenet_mnist_clang4(cluster, dirname, exes):
    skeleton_performance_lenet_mnist(cluster, dirname, exes, 'clang4')
    
def test_integration_performance_alexnet_clang4(cluster, dirname, exes, weekly):
  skeleton_performance_alexnet(cluster, dirname, exes, 'clang4', weekly)

def test_integration_performance_cache_alexnet_clang4(cluster, dirname, exes, weekly):
  skeleton_performance_cache_alexnet(cluster, dirname, exes, 'clang4', weekly)
                                        
def test_integration_performance_lenet_mnist_gcc4(cluster, dirname, exes):
  skeleton_performance_lenet_mnist(cluster, dirname, exes, 'gcc4')

def test_integration_performance_alexnet_gcc4(cluster, dirname, exes, weekly):
  skeleton_performance_alexnet(cluster, dirname, exes, 'gcc4', weekly)

def test_integration_performance_cache_alexnet_gcc4(cluster, dirname, exes, weekly):
  skeleton_performance_cache_alexnet(cluster, dirname, exes, 'gcc4', weekly)

def test_integration_performance_lenet_mnist_gcc7(cluster, dirname, exes):
  skeleton_performance_lenet_mnist(cluster, dirname, exes, 'gcc7')

def test_integration_performance_alexnet_gcc7(cluster, dirname, exes, weekly):
  skeleton_performance_alexnet(cluster, dirname, exes, 'gcc7', weekly)

def test_integration_performance_cache_alexnet_gcc7(cluster, dirname, exes, weekly):
  skeleton_performance_cache_alexnet(cluster, dirname, exes, 'gcc7', weekly)

def test_integration_performance_lenet_mnist_intel18(cluster, dirname, exes):
  skeleton_performance_lenet_mnist(cluster, dirname, exes, 'intel18')

def test_integration_performance_alexnet_intel18(cluster, dirname, exes, weekly):
  skeleton_performance_alexnet(cluster, dirname, exes, 'intel18', weekly)

def test_integration_performance_cache_alexnet_intel18(cluster, dirname, exes, weekly):
  skeleton_performance_cache_alexnet(cluster, dirname, exes, 'intel18', weekly)
